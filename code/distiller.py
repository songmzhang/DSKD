import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from utils import log_rank


class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student_model_type = args.model_type
        self.student_model, self.student_tokenizer = self.load_student_model()
        
        if self.args.teacher_model_path is not None:
            self.teacher_model, self.teacher_tokenizers = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizers = None, {}
        self.teacher_model_type = args.teacher_model_type

        if self.teacher_model and args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")
        
        if args.teacher_to_student_token_mapping is not None:
            self.tea2stu_token_mapping = json.load(open(args.teacher_to_student_token_mapping))
            log_rank(f"Load teacher-to-student token mapping from {args.teacher_to_student_token_mapping}")
        
        if args.teacher_to_student_id_mapping is not None:
            self.tea2stu_id_mapping = json.load(open(args.teacher_to_student_id_mapping))
            log_rank(f"Load teacher-to-student id mapping from {args.teacher_to_student_id_mapping}")

            self.stu2tea_id_mapping = {}
            for tea_id in self.tea2stu_id_mapping:
                if self.tea2stu_id_mapping[tea_id] not in self.stu2tea_id_mapping:
                    self.stu2tea_id_mapping[self.tea2stu_id_mapping[tea_id]] = [int(tea_id)]
                else:
                    self.stu2tea_id_mapping[self.tea2stu_id_mapping[tea_id]].append(int(tea_id))
            
            max_align_num = 1
            for stu_id in self.stu2tea_id_mapping:
                self.stu2tea_id_mapping[stu_id] = self.stu2tea_id_mapping[stu_id][:max_align_num] + \
                    [self.stu2tea_id_mapping[stu_id][-1]] \
                        * max(0, max_align_num - len(self.stu2tea_id_mapping[stu_id]))
                
            self.tea2stu_id_mapping = torch.LongTensor(list(self.tea2stu_id_mapping.values())).to(device)
            self.stu2tea_id_mapping_tea = torch.LongTensor(list(self.stu2tea_id_mapping.values())).to(device)
            self.stu2tea_id_mapping_stu = torch.LongTensor(list(self.stu2tea_id_mapping.keys())).to(device)

    @staticmethod
    def add_distiller_args(parser):
        group = parser.add_argument_group("distiller", "distiller configurations")
        group.add_argument("--projector-config-path", type=str, default=None,
                           help='path to projector_config.json')
        group.add_argument("--projector-path", type=str, default=None,
                           help='path to pretrained projector')
        group.add_argument("--projector-lr", type=float, default=0.001,
                           help='learning rate only for projection')
        group.add_argument("--pretrained-projector", type=str, default=None,
                           help='pretrained projector name')
        group.add_argument("--pretrained-projector-lr", type=float, default=0.001,
                           help='learning rate only for pretrained projector')
        group.add_argument("--vocab-alignment-path", type=str, default=None,
                           help='path for the vocab alignment file')
        group.add_argument("--teacher-to-student-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, teacher-to-student)')
        group.add_argument("--teacher-to-student-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, teacher-to-student)')
        group.add_argument("--student-to-teacher-token-mapping", type=str, default=None,
                           help='path for the vocab alignment file (token, student-to-teacher)')
        group.add_argument("--student-to-teacher-id-mapping", type=str, default=None,
                           help='path for the vocab alignment file (id, student-to-teacher)')
        return parser
    
    def load_tokenizer(self, model_type, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif model_type == "qwen":
            # tokenizer.pad_token_id = 151646
            tokenizer.eos_token_id = 151643
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return tokenizer

    def set_and_load_existing_projectors(self):
        self.projectors = nn.ModuleDict()
        projector_config = json.load(open(self.args.projector_config_path))
        name_dict = {
            "s": self.student_hidden_size, 
            "t": self.teacher_hidden_size,
            "relu": nn.ReLU()
        }
        # auto-parse projector config strings to construct nn.Module
        for projector_name in projector_config:
            # for d in projector_config[loc]:
            if projector_config[projector_name]["enabled"]:
                self.projectors[projector_name] = nn.Sequential()

                structure = projector_config[projector_name]["structure"].split("-")
                for i in range(len(structure)):
                    if structure[i] not in ["relu"]:
                        coef = 1 if not len(structure[i][:-1]) else int(structure[i][:-1])
                        base_size = name_dict[structure[i][-1]]
                        structure[i] = coef * base_size

                for i in range(len(structure) - 1):
                    if isinstance(structure[i], int) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(structure[i], structure[i+1])
                        )
                    elif isinstance(structure[i], int) and isinstance(structure[i+1], str):
                        self.projectors[projector_name].append(
                            name_dict[structure[i+1]]
                        )
                        last_size = structure[i]
                    elif isinstance(structure[i], str) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(last_size, structure[i+1])
                        )
                    else:
                        raise NotImplementedError(f"Invalid structure for '{structure}'")
                        
        # load existing projectors if already have
        self.load_existing_projectors()

    def load_existing_projectors(self):
        if self.args.projector_path is not None:
            projector_path = os.path.join(self.args.projector_path, "projector.pt")
        else:
            projector_path = os.path.join(self.args.model_path, "projector.pt")

        if os.path.exists(projector_path):
            projector_params = torch.load(projector_path, map_location=f"cuda:{self.device}")
            log_rank("Existing projector params: {}".format(list(projector_params.keys())))
            for key in self.projectors:
                try:
                    state_dict = {
                        n.split('.', 1)[1]: projector_params[n] for n in projector_params if n.startswith(key)
                    }
                    self.projectors[key].load_state_dict(state_dict)
                    log_rank("Load projector '{}' from current path.".format(key))
                except:
                    log_rank("Not compatible for projector '{}'".format(key))
                    continue
    
    def load_student_model(self):
        log_rank("Loading student model...")
        config = AutoConfig.from_pretrained(self.args.model_path, trust_remote_code=True)
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.args.model_type, self.args.model_path)
        
        if hasattr(config, "n_embed"):
            self.student_hidden_size = config.n_embed
        else:
            self.student_hidden_size = config.hidden_size
        
        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError("Invalid model_dtype for f`{self.args.model_dtype}`")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path, 
            config=config, 
            device_map=None, 
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        if self.args.peft is not None:
            if self.args.peft == "lora":
                model.enable_input_require_grads()
                if self.args.peft_path is not None:
                    if self.args.do_train:
                        _model = PeftModel.from_pretrained(model, self.args.peft_path)
                        state_dict = dict(_model.state_dict().items())
                        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM, 
                            inference_mode=(not self.args.do_train), 
                            r=self.args.peft_lora_r, 
                            lora_alpha=self.args.peft_lora_alpha, 
                            lora_dropout=self.args.peft_lora_dropout
                        )
                        model = get_peft_model(model, peft_config)
                        model.load_state_dict(state_dict)
                        del _model
                        del state_dict
                    else:
                        model = PeftModel.from_pretrained(model, self.args.peft_path)
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM, 
                        inference_mode=(not self.args.do_train), 
                        r=self.args.peft_lora_r, 
                        lora_alpha=self.args.peft_lora_alpha, 
                        lora_dropout=self.args.peft_lora_dropout
                    )
                    model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                raise NotImplementedError
        else:
            log_rank(' > number of parameters: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(self.args.teacher_model_path)
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.args.teacher_model_type, self.args.teacher_model_path)

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher_model_path, 
            config=config, 
            device_map=None, 
            torch_dtype=self.dtype,
            trust_remote_code=True
        )

        if self.args.peft is not None and self.args.teacher_peft_path is not None:
            if self.args.peft == "lora":
                model = PeftModel.from_pretrained(model, self.args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            log_rank(' > number of parameters of the teacher model: {:,}'.format(
                sum([p.nelement() for p in model.parameters()])
            ))
        for params in model.parameters():
            params.requires_grad = False
        return model, {self.args.teacher_model_type: tokenizer}
    
    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors"):
            if self.args.projector_lr:
                pretrained_proj = self.args.pretrained_projector.split(",") if self.args.pretrained_projector is not None else []
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b not in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.projector_lr
                })
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.args.pretrained_projector_lr
                })
            else:
                optimizer.add_param_group({
                    "params": [p for b in self.projectors for p in self.projectors[b].parameters()],
                })
        return optimizer

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch["output_batch"]
        loss, logging_output = criterion(
            self,
            input_data, 
            output_data,
            logging_output,
            loss_denom,
        )
        return loss, logging_output
