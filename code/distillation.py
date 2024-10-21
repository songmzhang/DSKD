import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import deepspeed
import shutil
import json
from tqdm import tqdm
import math
from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from transformers.integrations import HfDeepSpeedConfig
from arguments import get_args
from distiller import Distiller
from data_utils.distill_datasets import DistillDataset
from utils import (
    initialize,
    get_optimizer, 
    get_learning_rate_scheduler,
    print_rank, 
    log_rank,
    all_gather
)
from criterions import build_criterion
from rouge_metric import compute_metrics

torch.set_num_threads(4)


def prepare_dataset(args, distiller):
    data = {}
    if args.do_train:
        data["train"] = DistillDataset(
            args, "train", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of train data: {}".format(len(data["train"])))
        
        data["dev"] = DistillDataset(
            args, "dev", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of dev data: {}".format(len(data["dev"])))

        if os.path.exists(os.path.join(args.data_dir, "test.jsonl")):
            data["test"] = DistillDataset(
                args, "test", distiller.student_tokenizer,
                distiller.teacher_tokenizers
            )
            log_rank("Num of test data: {}".format(len(data["test"])))

    elif args.do_eval:
        data["test"] = DistillDataset(
            args, "test", distiller.student_tokenizer,
            distiller.teacher_tokenizers
        )
        log_rank("Num of test data: {}".format(len(data["test"])))
    else:
        raise ValueError("Do train and do eval must set one")
        
    return data


def finetune(
    args, 
    tokenizer: AutoTokenizer, 
    model: deepspeed.DeepSpeedEngine, 
    optimizer: AdamW, 
    lr_scheduler, 
    dataset, 
    device, 
):
    log_rank("Start Fine-tuning")
    start_time = time.time()

    if args.model_parallel:
        raise NotImplementedError
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        criterion = build_criterion(args)

    sampler = DistributedSampler(
        dataset["train"], 
        shuffle=True, 
        drop_last=True, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    train_dataloader = DataLoader(
        dataset['train'], 
        sampler=sampler, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset["train"].collate
    )
    
    step = 0
    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [], 
        "nll_loss": [],
        "kd_loss": [],
        "accuracy": [],
        "micro_step_time": [],
        "step_time": []
    }
    model_list = []

    # log_rank("Evaluate model before training...")
    # eval_loss, eval_results = evaluate(
    #     args, 
    #     tokenizer, 
    #     model.module.student_model, 
    #     dataset["dev"], 
    #     "dev", 
    #     device,
    #     repeat_times=1
    # )

    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)
        logging_output["epoch"] += 1
        log_rank("Start iterations of epoch {}".format(epoch + 1))
        model.train()
        end_epoch = False
        epoch_step = 0
        epoch_loss, epoch_nll_loss, epoch_kd_loss = 0.0, 0.0, 0.0
        train_iter = iter(train_dataloader)

        while True:
            # collect #gas batches first to calculate global batch size for token-level loss
            global_batch = []
            global_st_time = time.time()
            for i in range(args.gradient_accumulation_steps):
                try:
                    (input_batch, output_batch, _) = next(train_iter)
                    dataset["train"].move_to_device(
                        [input_batch, output_batch], device)
                    global_batch.append({
                        "input_batch": input_batch,
                        "output_batch": output_batch,
                    })
                except StopIteration:
                    end_epoch = True
                    break
            
            if end_epoch:
                break

            # get the true batch token num according to the whole batch (all_tok_num / (grad_acc * n_gpu))
            global_token_num = sum(
                batch["output_batch"]["label"].ne(-100).sum() for batch in global_batch)
            dist.all_reduce(global_token_num, dist.ReduceOp.SUM, group=dp_group)
            loss_denom = global_token_num / (args.gradient_accumulation_steps * dp_world_size)

            for batch in global_batch:
                st_time = time.time()
                loss, logging_output = model(
                    criterion, batch, logging_output, loss_denom)
                model.backward(loss)
                model.step()

                torch.cuda.synchronize()
                elapsed_time = time.time() - st_time
                logging_output["micro_step_time"].append(elapsed_time)
                step += 1

            logging_output["global_step"] += 1
            logging_output["step_time"].append(time.time() - global_st_time)
            epoch_step += 1

            def get_log(logging_output):
                logging_info = ""
                for key in logging_output:
                    if key == "epoch": continue
                    log_val = logging_output[key]
                    if isinstance(log_val, list) and len(log_val) > 0:
                        logging_info += f"{key}={sum(log_val) / len(log_val):.4f}, "
                    elif isinstance(log_val, int):
                        logging_info += f"{key}={log_val}, "
                    elif "lr" in key:
                        logging_info += f"{key}={log_val:.4e}, "
                
                log_rank("train | epoch {:0>3d}:   {:5d} / {:5d}  {}scale={:.4f}".format(
                    epoch + 1,
                    epoch_step,
                    args.train_iters_per_epoch,
                    logging_info,
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                ))

            if logging_output["global_step"] % args.log_interval == 0:
                logging_output["lr"] = lr_scheduler.get_last_lr()[0]
                if args.projector_config_path:
                    logging_output["projector_lr"] = lr_scheduler.get_last_lr()[-1]
                get_log(logging_output)
                epoch_loss += sum(logging_output["loss"])
                epoch_nll_loss += sum(logging_output["nll_loss"])
                epoch_kd_loss += sum(logging_output["kd_loss"])
                for key in logging_output:
                    if isinstance(logging_output[key], list):
                        logging_output[key] = []
            
        log_rank("End of epoch {}".format(epoch + 1))
        log_rank("train | epoch {:0>3d} | loss {:.4f} | nll_loss {:.4f} | kd_loss {:.4f}".format(
            epoch + 1,
            epoch_loss / (epoch_step * args.gradient_accumulation_steps),
            epoch_nll_loss / (epoch_step * args.gradient_accumulation_steps),
            epoch_kd_loss / (epoch_step * args.gradient_accumulation_steps),
        ))
        if args.save_dir and (epoch + 1) % args.save_interval == 0:
            if (epoch + 1) % args.eval_interval == 0:
                log_rank("Evaluating before saving model...")
                # Conduct evaluation before saving model
                eval_loss, eval_results = evaluate(
                    args, 
                    tokenizer, 
                    model.module.student_model, 
                    dataset["dev"], 
                    "dev", 
                    device
                )
                if "test" in dataset:
                    _, _ = evaluate(
                        args, 
                        tokenizer, 
                        model.module.student_model, 
                        dataset["test"], 
                        "test", 
                        device,
                        repeat_times=1
                    )

                if args.eval_gen:
                    ckpt_name = "epoch{}_step{}_loss{:.4f}_rougel{:.4f}".format(
                        epoch + 1, 
                        logging_output["global_step"], 
                        eval_loss, 
                        eval_results["rougeL"]
                    )
                else:
                    ckpt_name = "epoch{}_step{}_loss{:.4f}".format(
                        epoch + 1, 
                        logging_output["global_step"], 
                        eval_loss
                    )
                save_dir_path = os.path.join(args.save_dir, ckpt_name)
                
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    if not args.only_save_projector:
                        log_rank("Saving tokenizer...")
                        tokenizer.save_pretrained(save_dir_path)
                        log_rank("Saving model...")
                        model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    if hasattr(model.module, "projectors"):
                        log_rank("Saving projector...")
                        torch.save(
                            model.module.projectors.state_dict(), 
                            os.path.join(save_dir_path, "projector.pt")
                        )
                    # only keep best N checkpoints
                    if args.eval_gen:
                        model_list.append({
                            "path": save_dir_path, 
                            "score": eval_results["rougeL"]
                        })
                        model_list = sorted(model_list, key=lambda x: x["score"])
                    else:
                        model_list.append({
                            "path": save_dir_path, 
                            "score": eval_loss
                        })
                        model_list = sorted(model_list, key=lambda x: x["score"], reverse=True)
                        
                    if len(model_list) > args.keep_best_n_checkpoints:
                        removed_model = model_list.pop(0)
                        shutil.rmtree(removed_model["path"])

                    log_rank(f"Model has been saved to {save_dir_path}")
                dist.barrier()
            else:
                ckpt_name = "epoch{}_step{}".format(
                    epoch + 1, 
                    logging_output["global_step"], 
                )
                save_dir_path = os.path.join(args.save_dir, ckpt_name)
                
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    if not args.only_save_projector:
                        log_rank("Saving tokenizer...")
                        tokenizer.save_pretrained(save_dir_path)
                        log_rank("Saving model...")
                        model.module.student_model.save_pretrained(save_dir_path, safe_serialization=False)
                    if hasattr(model.module, "projectors"):
                        log_rank("Saving projector...")
                        torch.save(
                            model.module.projectors.state_dict(), 
                            os.path.join(save_dir_path, "projector.pt")
                        )
                    # only keep best N checkpoints
                    model_list.append({
                        "path": save_dir_path, 
                        "score": logging_output["global_step"]
                    })
                    model_list = sorted(model_list, key=lambda x: x["score"])
                        
                    if len(model_list) > args.keep_best_n_checkpoints:
                        removed_model = model_list.pop(0)
                        shutil.rmtree(removed_model["path"])

                    log_rank(f"Model has been saved to {save_dir_path}")
                dist.barrier()

    total_seconds = time.time() - start_time
    log_rank("Done training in {:0>2}:{:0>2}:{:0>2}".format(
        int(total_seconds // 3600), 
        int(total_seconds % 3600 // 60), 
        int(total_seconds % 60)
    ))

@torch.no_grad()
def evaluate(args, tokenizer, model, dataset, split, device, repeat_times=None):
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    loss_func = nn.CrossEntropyLoss(reduction="none")

    log_rank("Evaluating on {} set with {} GPU(s)".format(split, dp_world_size))

    if args.do_sample:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            no_repeat_ngram_size=args.no_repeat_ngram_size if split != "dev" else 0,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            min_length=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        repeat_times = args.eval_gen_repeat_times if repeat_times is None else repeat_times
    else:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            no_repeat_ngram_size=args.no_repeat_ngram_size if split != "dev" else 0,
            repetition_penalty=args.repetition_penalty,
            max_length=args.max_length,
            min_length=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )
        repeat_times = 1

    sampler = DistributedSampler(
        dataset, 
        shuffle=False, 
        drop_last=False, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        collate_fn=dataset.collate
    )

    model.eval()
    eval_info = {
        "loss": 0.0,
        "token_num": 0,
        "token_acc": 0.0,
        "top1_prob": 0.0,
    }
    all_loss = 0.0
    step = 0
    all_response_ids = [[] for i in range(repeat_times)]
    for input_batch, output_batch, gen_data in dataloader:
        dataset.move_to_device(
            [input_batch, output_batch, gen_data], device
        )
        logits = model(
            input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            position_ids=input_batch.get("position_ids", None)
        ).logits
        loss = loss_func(
            logits.view(-1, logits.shape[-1]), 
            output_batch["label"].view(-1)
        )
        pad_mask = output_batch["label"].ne(-100)
        token_num = pad_mask.sum()
        loss = loss.view_as(output_batch["label"]).masked_fill_(~pad_mask, 0.0).sum()
        token_acc_num = logits.argmax(-1).eq(output_batch["label"]).float()
        token_acc_num = token_acc_num.masked_fill_(~pad_mask, 0.0).sum()
        probs = logits.softmax(-1)
        top1_prob = probs.max(-1)[0].masked_fill(~pad_mask, 0.0).sum()

        dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(token_num, dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(token_acc_num, dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(top1_prob, dist.ReduceOp.SUM, group=dp_group)

        eval_info["loss"] += loss.item()
        eval_info["token_num"] += token_num.item()
        eval_info["token_acc"] += token_acc_num.item()
        eval_info["top1_prob"] += top1_prob.item()
    
    eval_info["loss"] /= eval_info["token_num"]
    eval_info["token_acc"] /= eval_info["token_num"]
    eval_info["top1_prob"] /= eval_info["token_num"]
    for key in eval_info:
        if isinstance(eval_info[key], float):
            eval_info[key] = round(eval_info[key], 6)
    
    eval_res = {}
    if args.eval_gen:
        # TODO: parallel sampling N different results using model.generate()
        for i in range(repeat_times):
            for input_batch, output_batch, gen_data in tqdm(
                dataloader, 
                desc=f"{i+1}-th evaluation: ", 
                disable=(dp_rank != 0 or not args.eval_tqdm)
            ):
                dataset.move_to_device([gen_data], device)
                max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
                try:
                    gen_out = model.generate(
                        **gen_data,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens
                    )
                except:
                    model = model.float()
                    gen_out = model.generate(
                        **gen_data,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens
                    )
                    model = model.half()
                
                full_ids = gen_out.sequences
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids[i].append(response_ids)
            
            all_response_ids[i] = torch.cat(all_response_ids[i], dim=0)
            all_response_ids[i] = all_gather(
                all_response_ids[i], 
                dim=1, 
                world_size=dp_world_size, 
                group=dp_group, 
                op="stack"
            )
            all_response_ids[i] = all_response_ids[i].view(
                -1, 
                all_response_ids[i].size(-1)
            )
            responses = tokenizer.batch_decode(
                all_response_ids[i], 
                skip_special_tokens=True
            )
            references = dataset.answers
            responses = responses[:len(references)]
            res = compute_metrics(responses, references)
            log_rank("eval_results in run@{}: {}".format(i + 1, res))
            
            for key in res:
                if key in eval_res:
                    eval_res[key].append(res[key])
                else:
                    eval_res[key] = [res[key]]
        
        for key in eval_res:
            eval_res[key] = round(sum(eval_res[key]) / len(eval_res[key]), 4)
    
    log_str = f"{split} | {eval_info} | {eval_res}"
    log_rank(log_str)
    model.train()
    return eval_info["loss"], eval_res


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    # save arguments
    if dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30)
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]
    if "bf16" in ds_config:
        args.fp32 = not ds_config["bf16"]["enabled"]
    log_rank(args)
    args.deepspeed_config = None
    
    # prepare for deepspeed ZeRO-3
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    
    log_rank("Initializing a distiller for knowledge distillation...")
    distiller = Distiller(args, device)
    dataset = prepare_dataset(args, distiller)
    
    dp_world_size = dist.get_world_size()
    
    if args.do_train:
        args.train_iters_per_epoch = int(
            len(dataset["train"]) / 
            (args.batch_size * dp_world_size * args.gradient_accumulation_steps)
        )
        log_rank("Train iters per epoch = {}".format(args.train_iters_per_epoch))

        assert args.total_iters is not None or args.num_epochs is not None
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.num_epochs
        if args.num_epochs is None:
            args.num_epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        log_rank("Total_iters = {}".format(args.total_iters))
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    optimizer = get_optimizer(args, distiller.student_model)
    optimizer = distiller.add_optimizer_param_group(optimizer)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    if args.do_train:
        finetune(args, distiller.student_tokenizer, model, optimizer, lr_scheduler, dataset, device)
   
    if args.do_eval:
        evaluate(
            args, 
            distiller.student_tokenizer, 
            model, 
            dataset["test"], 
            "test", 
            0, 
            device
        )
        
    
if __name__ == "__main__":
    main()
