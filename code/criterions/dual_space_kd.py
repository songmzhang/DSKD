import torch
from .various_divergence import VariousDivergence


class DualSpaceKD(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        
        kd_loss, log = self.compute_dual_space_kd_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output

    def compute_dual_space_kd_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        target = output_data["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        # student space
        t2s_hiddens = distiller.projectors["t2s"](teacher_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        t2s_kd_loss = (t2s_kd_loss * pad_mask).sum()

        # teacher space
        s2t_hiddens = distiller.projectors["s2t"](hiddens)
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        s2t_kd_loss = self.compute_forward_kl_divergence(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
        
        kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss

        t2s_acc = (t2s_logits.argmax(-1).eq(target) * pad_mask).sum()
        s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()

        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_acc"] = t2s_acc
        log["s2t_acc"] = s2t_acc
        log["kd_loss"] = kd_loss
        return kd_loss, log
    
    