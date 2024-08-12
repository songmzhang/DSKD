import logging
import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
import transformers
import editdistance
from typing import Dict, List
from .various_divergence import VariousDivergence


TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.LlamaTokenizerFast: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.GPT2Tokenizer: "Ġ",
    transformers.GPT2TokenizerFast: "Ġ",
    transformers.Qwen2Tokenizer: "Ġ",
    transformers.Qwen2TokenizerFast: "Ġ",
}

class MinEditDisForwardKLD(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super(MinEditDisForwardKLD, self).__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
    
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
            
        teacher_logits = self.get_aligned_teacher_logits(
            logits, 
            teacher_outputs.logits, 
            input_data,
            output_data,
            distiller
        )
        
        kd_loss = self.compute_forward_kl_divergence(
            logits, 
            teacher_logits, 
            output_data["label"],
            log=log
        )

        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, 
            output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            log
        )
        return loss / batch_denom, logging_output

    def get_aligned_teacher_logits(
        self, logits, teacher_logits, input_data, output_data, distiller,
    ):
        target = output_data["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        target_ids = torch.where(
            pad_mask, 
            target, 
            torch.ones_like(target) * distiller.student_tokenizer.eos_token_id
        )
        stu_tokenizer = distiller.student_tokenizer
        tea_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]

        bsz = target.shape[0]
        aligned_tea_logits = []
        for i in range(bsz):
            stu_content_idx = torch.nonzero(target[i].ne(self.padding_id)).view(-1)
            stu_input_ids = input_data["input_ids"][i, stu_content_idx]
            stu_target_ids = target_ids[i, stu_content_idx]

            tea_content_idx = torch.nonzero(teacher_target[i].ne(self.padding_id)).view(-1)
            tea_input_ids = input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][i, tea_content_idx]

            stu_per_step_logits = logits[i, stu_content_idx, :].float()
            tea_per_step_logits = teacher_logits[i, tea_content_idx, :].float()   # [slen, vocab]

            aligned_tea_content_per_step_logits = self.transform_step_logits_fast(
                stu_tokenizer,
                tea_tokenizer,
                stu_input_ids,
                stu_per_step_logits,
                stu_target_ids,
                tea_input_ids,
                tea_per_step_logits,
                blending_to_base_mapping=distiller.tea2stu_id_mapping,
                base_to_blending_mapping_blending_ids=distiller.stu2tea_id_mapping_tea,
                base_to_blending_mapping_base_ids=distiller.stu2tea_id_mapping_stu
            )

            aligned_tea_per_step_logits = logits[i].float().detach()
            aligned_tea_per_step_logits[stu_content_idx] = aligned_tea_content_per_step_logits
            aligned_tea_logits.append(aligned_tea_per_step_logits)
        
        aligned_tea_logits = torch.stack(aligned_tea_logits, 0)
        return aligned_tea_logits

    def transform_step_logits_fast(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_input_ids: torch.LongTensor,
        base_model_per_step_logits: torch.FloatTensor,
        base_model_target_ids: torch.LongTensor,
        blending_model_input_ids: torch.LongTensor,
        blending_model_per_step_logits: torch.FloatTensor,
        blending_to_base_mapping: torch.LongTensor = None,
        base_to_blending_mapping_blending_ids: torch.LongTensor = None,
        base_to_blending_mapping_base_ids: torch.LongTensor = None,
        device: str = None,
    ):
        """faster implementation to align logits"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            base_model_tokenizer.__class__
        ]
        blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            blending_model_tokenizer.__class__
        ]

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        # obtain sequence token alignment (each stu token to which tea token)
        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        ) 
        unalign_mask = [1 if len(a) == 1 else 0 for a in base_to_blending]
        unalign_mask = torch.tensor(unalign_mask).to(base_model_input_ids.device)

        # for one-to-one mapping, align their logits; for one-to-many mapping, use ground-truth one-hot target
        base_to_blending = [a[0] if len(a) == 1 else 0 for a in base_to_blending]
        base_to_blending = torch.LongTensor(base_to_blending).to(base_model_input_ids.device)
        # for one-to-one mapping, ensure they are really similar
        unalign_mask = unalign_mask & base_model_input_ids.eq(blending_to_base_mapping[blending_model_input_ids[base_to_blending]])
        # get the logits of mapped tea tokens
        blending_model_per_step_logits = blending_model_per_step_logits[base_to_blending]
        blending_model_per_step_logits = blending_model_per_step_logits[
            :, base_to_blending_mapping_blending_ids.view(-1)
        ]
        blending_model_per_step_logits = blending_model_per_step_logits.view(
            -1, 
            base_to_blending_mapping_blending_ids.shape[0], 
            base_to_blending_mapping_blending_ids.shape[1]
        ).max(-1)[0]
        # transform teacher logits to student logits
        blending_to_base_logits = torch.ones_like(base_model_per_step_logits) * (-100000)
        blending_to_base_logits[:, base_to_blending_mapping_base_ids] = blending_model_per_step_logits
        
        unalign_mask = unalign_mask \
                     & blending_to_base_logits.max(-1)[0].ne(-100000)
        # mask unaligned position, use ground-truth target (one-hot)
        one_hot_logits = F.one_hot(base_model_target_ids, num_classes=base_model_per_step_logits.shape[-1])
        one_hot_logits = (1 - one_hot_logits) * (-100000) + (one_hot_logits) * 100
        
        unalign_mask = unalign_mask.unsqueeze(-1)
        blending_to_base_logits = torch.where(
            unalign_mask.repeat(1, base_model_per_step_logits.shape[-1]).eq(1),
            blending_to_base_logits,
            one_hot_logits
        )

        return blending_to_base_logits


    def transform_step_logits(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_vocab: Dict[str, int],
        base_model_input_ids: List[int],
        blending_model_input_ids: List[int],
        blending_model_per_step_logits: List[List[float]],
        blending_model_per_step_indices: List[List[int]],
        vocab_align_type: str = "hard",
        blending_to_base_mapping: Dict[str, str] = None,
    ):
        """Align blending model per step logits & indices with base model. (original implementation in FuseLLM)"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            base_model_tokenizer.__class__
        ]
        blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            blending_model_tokenizer.__class__
        ]

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        )
        aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = (
            [],
            [],
        )
        for i, blending_idx in enumerate(base_to_blending):
            aligned_blending_model_per_step_logit = []
            aligned_blending_model_per_step_index = []
            if len(blending_idx) == 1:  # one base token map to one blending token
                j = blending_idx[0]
                base_token = base_model_tokens[i]
                blending_token = blending_model_tokens[j].replace(
                    blending_model_special_token, base_model_special_token
                )
                if (
                    (
                        blending_model_tokenizer.__class__
                        == transformers.GPTNeoXTokenizerFast
                        or blending_model_tokenizer.__class__
                        == transformers.GPT2TokenizerFast
                    )
                    and i == 0
                    and base_token.startswith(base_model_special_token)
                    and not blending_token.startswith(base_model_special_token)
                ):
                    blending_token = (
                        base_model_special_token + blending_token
                    )  # special case for mpt
                if vocab_align_type == "hard":
                    if (
                        base_token == blending_token
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                elif vocab_align_type == "soft":
                    if (base_token == blending_token) or (
                        blending_token in blending_to_base_mapping
                        and base_token == blending_to_base_mapping[blending_token]
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):  
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            blending_t = blending_to_base_mapping[blending_t]
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                            else:
                                logging.warning(
                                    f"blending_t: {blending_t} not in base_model_vocab!"
                                )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                else:
                    logging.warning(
                        f"The vocab_align_type: '{vocab_align_type}' is not support!"
                    )
                    raise NotImplementedError
            else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            aligned_blending_model_per_step_indices.append(
                aligned_blending_model_per_step_index
            )
            aligned_blending_model_per_step_logits.append(
                aligned_blending_model_per_step_logit
            )
        return (
            aligned_blending_model_per_step_logits,
            aligned_blending_model_per_step_indices,
        )
    
    def dtw(self, series_1, series_2, norm_func=np.linalg.norm):
        """Use dynamic time wrapping to align to tokenizers, modified from:
        https://github.com/talcs/simpledtw/blob/master/simpledtw.py"""
        matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        for i, vec1 in enumerate(series_1):
            for j, vec2 in enumerate(series_2):
                cost = norm_func(vec1, vec2)
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                )
        matrix = matrix[1:, 1:]
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        matches = []
        mappings_series_1 = [list() for v in range(matrix.shape[0])]
        mappings_series_2 = [list() for v in range(matrix.shape[1])]
        while i > 0 or j > 0:
            matches.append((i, j))
            mappings_series_1[i].append(j)
            mappings_series_2[j].append(i)
            option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
            option_up = matrix[i - 1, j] if i > 0 else np.inf
            option_left = matrix[i, j - 1] if j > 0 else np.inf
            move = np.argmin([option_diag, option_up, option_left])
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        matches.append((0, 0))
        mappings_series_1[0].append(0)
        mappings_series_2[0].append(0)
        matches.reverse()
        for mp in mappings_series_1:
            mp.reverse()
        for mp in mappings_series_2:
            mp.reverse()

        return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
