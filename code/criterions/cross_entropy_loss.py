import torch
import torch.nn as nn
import torch.distributed as dist


class CrossEntropyLoss(nn.Module):
    def __init__(self, args, padding_id=-100) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.label_smoothing = args.label_smoothing
        self.padding_id = padding_id
    
    def forward(self, distiller, input_data, output_data, logging_output, batch_denom):
        self.distiller = distiller
        model = distiller.student_model
        logits = model(**input_data).logits
        loss, nll_loss = self.compute_cross_entropy_loss(logits, output_data["label"])

        accuracy = self.compute_token_accuracy(
            logits, 
            output_data["label"], 
        )
        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            {
                "loss": loss,
                "nll_loss": nll_loss,
                "accuracy": accuracy
            }
        )
        return loss / batch_denom, logging_output

    def compute_cross_entropy_loss(self, logits, target, log=None):
        pad_mask = target.ne(self.padding_id)
        target = target.unsqueeze(-1)
        target = torch.where(
            target.eq(-100), 
            torch.zeros_like(target),
            target
        )
        logits = logits.masked_fill_(logits.isnan() | logits.isinf(), 0.0)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        nll_loss = -lprobs.gather(-1, target).squeeze(-1)
        nll_loss = (nll_loss * pad_mask).sum()
        
        if self.label_smoothing > 0:
            eps_i = self.label_smoothing / (lprobs.shape[-1] - 1)
            smooth_loss = -lprobs.sum(-1)
            loss = (1 - self.label_smoothing - eps_i) * nll_loss + eps_i * smooth_loss
            loss = (loss * pad_mask).sum()
        else:
            loss = nll_loss
        
        if log is not None:
            log["nll_loss"] = nll_loss
        
        return loss, nll_loss

    def compute_token_accuracy(self, logits, target):
        pad_mask = target.ne(self.padding_id)
        acc = (logits.argmax(-1).eq(target) * pad_mask).sum()
        return acc
    
    def record_logits(self, logits, target, log, teacher_logits=None, teacher_target=None):
        pad_mask = target.eq(self.padding_id)
        pos_target = torch.where(
            pad_mask,
            torch.zeros_like(target),
            target
        )
        target_logits = logits.gather(-1, pos_target.unsqueeze(-1)).squeeze(-1)
        non_target_logits = (logits.sum(-1) - target_logits) / (logits.shape[-1] - 1)
        target_logits = target_logits.masked_fill_(pad_mask, 0.0).sum()
        non_target_logits = non_target_logits.masked_fill_(pad_mask, 0.0).sum()
        
        log["target_logits"] = target_logits
        log["non_target_logits"] = non_target_logits

        if teacher_logits is not None:
            assert teacher_target is not None
            teacher_pad_mask = teacher_target.eq(self.padding_id)
            pos_teacher_target = torch.where(
                teacher_pad_mask,
                torch.zeros_like(teacher_target),
                teacher_target
            )
            teacher_target_logits = teacher_logits.gather(-1, pos_teacher_target.unsqueeze(-1)).squeeze(-1)
            teacher_non_target_logits = (teacher_logits.sum(-1) - teacher_target_logits) / (teacher_logits.shape[-1] - 1)
            teacher_target_logits = teacher_target_logits.masked_fill_(teacher_pad_mask, 0.0).sum()
            teacher_non_target_logits = teacher_non_target_logits.masked_fill_(teacher_pad_mask, 0.0).sum()
            
            log["teacher_target_logits"] = teacher_target_logits
            log["teacher_non_target_logits"] = teacher_non_target_logits
    
    def record_logging_output(self, logging_output, batch_denom, content):
        for k, v in content.items():
            record_v = v / batch_denom
            dist.all_reduce(record_v, dist.ReduceOp.SUM)
            record_v = record_v.item() / dist.get_world_size()
            if k in logging_output:
                logging_output[k].append(record_v)
            else:
                logging_output[k] = [record_v]
        return logging_output