import torch
from .cross_entropy_loss import CrossEntropyLoss


class UniversalLogitDistillation(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
    
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
                output_hidden_states=True
            )
        
        kd_loss, log = self.compute_universal_logit_distillation_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )

        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        # self.record_logits(
        #     outputs.logits, 
        #     output_data["label"], 
        #     log, 
        #     teacher_outputs.logits, 
        #     output_data[f"teacher_{distiller.teacher_model_type}_label"]
        # )

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output

    def compute_universal_logit_distillation_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        student_target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        # align the start of the student&teacher sequences
        for i in range(student_target.shape[0]):
            stu_start_idx = student_target[i].ne(self.padding_id).nonzero()[0][0]
            tea_start_idx = teacher_target[i].ne(self.padding_id).nonzero()[0][0]
            student_target[i] = torch.cat([
                student_target[i][stu_start_idx:], 
                student_target[i][:stu_start_idx]], dim=0
            )
            student_logits[i] = torch.cat([
                student_logits[i][stu_start_idx:, :],
                student_logits[i][:stu_start_idx, :]], dim=0
            )
            teacher_target[i] = torch.cat([
                teacher_target[i][tea_start_idx:], 
                teacher_target[i][:tea_start_idx]], dim=0
            )
            teacher_logits[i] = torch.cat([
                teacher_logits[i][tea_start_idx:, :],
                teacher_logits[i][:tea_start_idx, :]], dim=0
            )
        
        student_probs = torch.softmax(student_logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        sorted_student_probs = student_probs.sort(-1, descending=True).values
        sorted_teacher_probs = teacher_probs.sort(-1, descending=True).values

        vocab_size_gap = sorted_student_probs.shape[-1] - sorted_teacher_probs.shape[-1]
        bsz, slen = sorted_student_probs.shape[0], sorted_student_probs.shape[1]
        if vocab_size_gap > 0:
            sorted_teacher_probs = torch.cat([
                sorted_teacher_probs, 
                torch.zeros(bsz, slen, vocab_size_gap).to(teacher_probs)], 
                dim=-1
            )
        elif vocab_size_gap < 0:
            sorted_student_probs = torch.cat([
                sorted_student_probs, 
                torch.zeros(bsz, slen, -vocab_size_gap).to(student_probs)], 
                dim=-1
            )
        
        uld_loss = (sorted_student_probs - sorted_teacher_probs).abs().sum(-1)
        pad_mask = student_target.ne(self.padding_id) & teacher_target.ne(self.padding_id)
        uld_loss = (uld_loss * pad_mask).sum()
        log["uld_loss"] = uld_loss
        return uld_loss, log
    