import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
from tqdm import tqdm


save_path = "../../figures/structure_dist/train_"
if not os.path.exists(save_path):
    os.mkdir(save_path)


device = "cuda:1"

vanilla_kd_path = "vanilla_kd_ckpt_path"
dskd_path = "dskd_ckpt_path"
sft_path = "sft_ckpt_path"
teacher_path = "teacher_ckpt_path"

vanilla_kd_model = AutoModelForCausalLM.from_pretrained(vanilla_kd_path).to(device)
dskd_model = AutoModelForCausalLM.from_pretrained(dskd_path).to(device)
no_kd_model = AutoModelForCausalLM.from_pretrained(sft_path).to(device)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_path).to(device)

tokenizer = AutoTokenizer.from_pretrained(sft_path)

with open("../../data/dolly/train.jsonl") as f:
    data = [json.loads(s) for s in f.readlines()[:1000]]

prompt = [d["prompt"][:256] for d in data]
output = [d["output"][:256] for d in data]

prompt_inputs = [tokenizer(text, return_tensors="pt") for text in prompt]
output_inputs = [tokenizer(text, return_tensors="pt") for text in output]

def cal_all_sim(model, teacher_model):
    all_cosine_dist = []
    all_innerprod_dist = []
    for pinp, oinp in tqdm(list(zip(prompt_inputs, output_inputs))):
        inp = {}
        for key in pinp:
            inp[key] = torch.cat([pinp[key], oinp[key]], 1)
        
        inp["position_ids"] = torch.tensor([list(range(inp["input_ids"].shape[1]))])

        for x in inp:
            inp[x] = inp[x].to(device)
        
        prompt_len = pinp["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inp, output_hidden_states=True)
            hiddens = outputs.hidden_states[-1][:, prompt_len:]
        
            teacher_outputs = teacher_model(**inp, output_hidden_states=True)
            teacher_hiddens = teacher_outputs.hidden_states[-1][:, prompt_len:]
            
        norm_hiddens = hiddens / hiddens.norm(p=2, dim=-1, keepdim=True)
        stu_self_cosine = norm_hiddens.matmul(norm_hiddens.transpose(-1, -2))
        stu_self_innerprod = hiddens.matmul(hiddens.transpose(-1, -2))
        stu_self_innerprod = stu_self_innerprod / stu_self_innerprod.sum(-1, keepdim=True)

        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.norm(p=2, dim=-1, keepdim=True)
        tea_self_cosine = norm_teacher_hiddens.matmul(norm_teacher_hiddens.transpose(-1, -2))
        tea_self_innerprod = teacher_hiddens.matmul(teacher_hiddens.transpose(-1, -2))
        tea_self_innerprod = tea_self_innerprod / tea_self_innerprod.sum(-1, keepdim=True)

        cosine_sim = (stu_self_cosine - tea_self_cosine).abs().mean()
        innerprod_sim = (stu_self_innerprod - tea_self_innerprod).abs().sum(-1).mean()
        all_cosine_dist.append(cosine_sim.cpu().item())
        all_innerprod_dist.append(innerprod_sim.cpu().item())

    return all_cosine_dist, all_innerprod_dist

no_kd_cosine_sim, no_kd_innerprod_sim = cal_all_sim(no_kd_model, teacher_model)
vanilla_kd_cosine_sim, vanilla_kd_innerprod_sim = cal_all_sim(vanilla_kd_model, teacher_model)
dskd_cosine_sim, dskd_innerprod_sim = cal_all_sim(dskd_model, teacher_model)

plt.boxplot(
    [no_kd_cosine_sim, vanilla_kd_cosine_sim, dskd_cosine_sim], 
    labels=["SFT", "Vanilla KD", "DSKD"],
    showfliers=False,
    showmeans=False
)
plt.grid(axis="y", linestyle=":")
plt.xlabel("Methods")
plt.ylabel("Representation Distance (Cosine)")
plt.savefig(save_path + "cosine.png")
# plt.savefig(save_path + "cosine.pdf")

plt.cla()
plt.boxplot(
    [no_kd_innerprod_sim, vanilla_kd_innerprod_sim, dskd_innerprod_sim], 
    labels=["SFT", "Vanilla KD", "DSKD"], 
    showfliers=False,
    showmeans=False
)
plt.grid(axis="y", linestyle=":")
plt.ylabel("Representation Distance (Inner Product)")
plt.savefig(save_path + "attn.png")
# plt.savefig(save_path + "attn.pdf")