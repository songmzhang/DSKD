# Dual-Space Knowledge Distillation for Large Language Models

<small>Songming Zhang, Xue Zhang, Zengkui Sun, Yufeng Chen*, Jinan Xu</small>

<a href="https://arxiv.org/abs/2406.17328"><img src="https://img.shields.io/badge/Paper-arXiv:2406.17328-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a>

Some of our code follows [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) and [Distillm](https://github.com/jongwooko/distillm/tree/master).

## News
- \[2024.10.21\] Our code has supported the distillation from a **72B** model to a 1.5B model with DeepSpeed ZeRO-3.
- \[2024.09.21\] Our paper has been accepted by the main conference of EMNLP 2024！🥳🥳

## Requirements
- deepspeed >= 0.14.0
- torch >= 2.0.1
- transformers >= 4.40.2
- peft >= 0.8.2
- rouge_score >= 0.1.2

## Data
The processed data used in our paper can be downloaded [here](https://drive.google.com/drive/folders/1ZUsNVgWevACV9D-AHVNi9C7PX_2itzb8?usp=sharing).

## Models
You can download the corresponding model files (e.g., `pytorch_model.bin` or `model.safetensors`) of LLMs used in this paper into `model_hub/*/*/`.

Here are the links of these models on huggingface:
- GPT2-120M: [Here](https://huggingface.co/openai-community/gpt2)
- GPT2-1.5B (trained on Dolly by Gu et al.): [Here](https://github.com/microsoft/LMOps/blob/main/minillm/README.md#31-resources)
- Qwen1.5-1.8B: [Here](https://huggingface.co/Qwen/Qwen1.5-1.8B)
- TinyLLaMA-1.1B: [Here](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- Llama2-7B: [Here](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Mistral-7B: [Here](https://huggingface.co/mistralai/Mistral-7B-v0.1)

## Training
### SFT for teacher models
For Qwen1.5-1.8B (full fine-tuning), run:
```bash
bash scripts/gpt2/sft_teacher_qwen.sh
```

For LLaMA2-7B (LoRA), run:
```bash
bash scripts/tinyllama/sft_teacher_llama2.sh
```

For Mistral-7B (LoRA), run:
```bash
bash scripts/tinyllama/sft_teacher_mistral.sh
```

### SFT for student models
For GPT2-base (full fine-tuning), run:
```bash
bash scripts/gpt2/sft_gpt2_base.sh
```

For TinyLLaMA-1.1B (LoRA), run:
```bash
bash scripts/tinyllama/sft_tinyllama.sh
```

P.S. You may encounter an error **when directly loading the model checkpoint of TinyLLaMA**. This is because of the mismatched versions of `transformers` between TinyLLaMA suggested (4.31) and the one you use.
A concise solution to fix this can be referred to in [this issue](https://github.com/songmzhang/DSKD/issues/8).

### KD for the Same Vocabulary
#### Vanilla KD framework
For GPT2-base, run:
```bash
bash scripts/gpt2/vanilla_kd_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/vanilla_kd_tinyllama.sh
```

You can change the distance functions (e.g., KL Divergence, Reverse KL Divergence, JS Divergence, etc.) using `KD_OBJ` in the above scripts.

#### Dual-Space KD framework
For GPT2-base, run:
```bash
bash scripts/gpt2/dskd_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/dskd_tinyllama.sh
```

Also, you can change the distance functions using `KD_OBJ` in the above scripts.

### KD for different vocabularies
#### Logits Alignment by Minimum Edit Distance ([paper](https://arxiv.org/abs/2401.10491), [original implementation](https://github.com/fanqiwan/FuseAI))
The original implementation in this [repo](https://github.com/fanqiwan/FuseAI) pre-processes the logit alignment before distillation, while we re-implement this method by faster calculating alignment during distillation in [code/criterions/min_edit_dis_kld.py](https://github.com/songmzhang/DSKD/blob/1fc215196ea473aab971eea3b765ade57bbfb21b/code/criterions/min_edit_dis_kld.py).

For GPT2-base, run:
```bash
bash scripts/gpt2/minedit_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/minedit_tinyllama.sh
```

#### Universal Logit Distillation ([paper](https://arxiv.org/abs/2402.12030), [original implementation](https://github.com/Nicolas-BZRD/llm-recipes))
We also re-implement this method in [code/criterions/universal_logit_distillation.py](https://github.com/songmzhang/DSKD/blob/1fc215196ea473aab971eea3b765ade57bbfb21b/code/criterions/universal_logit_distillation.py).

For GPT2-base, run:
```bash
bash scripts/gpt2/uld_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/uld_tinyllama.sh
```

#### Our Dual-Space KD with Cross-Model Attention (CMA)
For GPT2-base, run:
```bash
bash scripts/gpt2/dskd_cma_gpt2_base.sh
```

For TinyLLaMA-1.1B, run:
```bash
bash scripts/tinyllama/dskd_cma_tinyllama.sh
```

### File Structures in Output Directory
The output directory will be created under `./outputs` automatically after you run the training scripts. 
For full fine-tuning, the file structure of the output directory is as follows (take gpt2 SFT as an example):
```
./outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── config.json
│   └── pytorch_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```
For LoRA fine-tuning, the file structure of the output directory is as follows (take TinyLLaMA LoRA SFT as an example):
```
./outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../
│
├── epochA_step... (model files of epoch A, you can directly load it by AutoModelForCausalLM.from_pretrained(this path))/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
├── epochB_step... (only exists when SAVE_BEST_N_CKPTS >= 2, similar to epochA_.../)/
│   ├── adapter_config.json
│   └── adapter_model.bin
│   └── tokenizer.json
│   └── ...
│
└── ...
│
└── args.json (The arguments of training)
│
└── train.log (Training log)
```

## Evaluation
### Evaluate Full Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval.sh ${CKPT_PATH} ${EVAL_BATCH_SIZE}
```
According to the above structure, `CKPT_PATH` is the **absolute path** of the model files like `/home/xxx/DSKD/outputs/gpt2/gpt2-base/sft/criterion=cross_entropy__default-bf16__.../epochA_step...`.

### Evaluate LoRA Fine-tuning Checkpoints
```bash
bash scripts/eval/run_eval_lora.sh ${LORA_ADAPTER_PATH} ${EVAL_BATCH_SIZE}
```
Please note that `MODEL_PATH` in `run_eval_lora.sh` should be changed for different base models (TinyLLaMA, LLaMA2, Mistral).

Similarly, `LORA_ADAPTER_PATH` is the **absolute path** of the LoRA adapter files like `/home/xxx/DSKD/outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=cross_entropy__lora-rank=256-alpha=8.../epochA_step...`.

## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@article{zhang2024dskd,
      title={Dual-Space Knowledge Distillation for Large Language Models}, 
      author={Songming Zhang and Xue Zhang and Zengkui Sun and Yufeng Chen and Jinan Xu},
      year={2024},
      journal={arXiv preprint arXiv:2406.17328},
}
```
