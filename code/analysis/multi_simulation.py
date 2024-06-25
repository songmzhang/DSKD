import torch
from matplotlib import pyplot as plt
from tqdm import trange
import torch.nn as nn
from torch.optim import SGD
import os


simulation_times = 100
data_num = 100
cls_num = 10000
lr =1.0
# device = "cpu"
device = "cuda:1"
obj = "kl"
save_path = f"../../figures/{obj}/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

class Net(nn.Module):
    def __init__(self, data_num, cls_num, device, obj):
        super(Net, self).__init__()
        
        self.h1 = nn.Parameter(torch.randn(data_num, 2, requires_grad=True, device=device))
        self.h1.data = self.h1.data + 3
        self.e1 = nn.Parameter(torch.randn(cls_num, 2, requires_grad=True, device=device))

        self.h2 = nn.Parameter(torch.randn(data_num, 2, requires_grad=True, device=device))
        self.e2 = nn.Parameter(torch.randn(cls_num, 2, requires_grad=True, device=device))
        self.h2.data = self.h1.data.clone()
        self.e2.data = self.e1.data.clone()

        self.h0 = torch.randn(data_num, 2)
        self.h0.data = self.h1.data.clone()

        self.h3 = torch.randn(data_num, 2, device=device) * 2
        self.e3 = torch.randn(cls_num, 2, device=device)

        self.obj = obj

    def cal_kl(self, l1, l2):
        lprobs1 = torch.log_softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        lprobs2 = torch.log_softmax(l2, -1)
        kl = (probs2 * (lprobs2 - lprobs1)).sum(-1)
        loss = kl.mean()
        return loss

    def cal_rkl(self, l1, l2):
        lprobs1 = torch.log_softmax(l1, -1)
        probs1 = torch.softmax(l1, -1)
        lprobs2 = torch.log_softmax(l2, -1)
        rkl = (probs1 * (lprobs1 - lprobs2)).sum(-1)
        loss = rkl.mean()
        return loss
    
    def cal_js(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        mprobs = (probs1 + probs2) / 2
        lprobs1 = torch.log(probs1 + 1e-9)
        lprobs2 = torch.log(probs2 + 1e-9)
        lmprobs = torch.log(mprobs + 1e-9)
        kl1 = probs1 * (lprobs1 - lmprobs)
        kl2 = probs2 * (lprobs2 - lmprobs)
        js = (kl1 + kl2) / 2
        loss = js.sum(-1).mean()
        return loss

    def cal_skl(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        probs1 = 0.9 * probs1 + 0.1 * probs2
        lprobs1 = torch.log(probs1 + 1e-9)
        lprobs2 = torch.log(probs2 + 1e-9)
        kl = probs2 * (lprobs2 - lprobs1)
        loss = kl.sum(-1).mean()
        return loss

    def cal_srkl(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        probs2 = 0.9 * probs2 + 0.1 * probs1
        lprobs1 = torch.log(probs1 + 1e-9)
        lprobs2 = torch.log(probs2 + 1e-9)
        kl = probs1 * (lprobs1 - lprobs2)
        loss = kl.sum(-1).mean()
        return loss

    def cal_akl(self, l1, l2):
        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        sorted_probs2, sorted_idx = probs2.sort(-1)
        sorted_probs1 = probs1.gather(-1, sorted_idx)
        gap = (sorted_probs2 - sorted_probs1).abs()
        cum_probs2 = torch.cumsum(sorted_probs2, -1)
        tail_mask = cum_probs2.lt(0.5).float()
        g_head = (gap * (1 - tail_mask)).sum(-1).detach()
        g_tail = (gap * tail_mask).sum(-1).detach()

        probs1 = torch.softmax(l1, -1)
        probs2 = torch.softmax(l2, -1)
        lprobs1 = torch.log_softmax(l1, -1)
        lprobs2 = torch.log_softmax(l2, -1)
        fkl = (probs2 * (lprobs2 - lprobs1)).sum(-1)
        rkl = (probs1 * (lprobs1 - lprobs2)).sum(-1)

        akl = (g_head / (g_head + g_tail)) * fkl + (g_tail / (g_head + g_tail)) * rkl
        loss = akl.mean()
        return loss

    def forward(self, share_head):
        if not share_head:
            stu_logits = self.h1.matmul(self.e1.transpose(-1, -2))
            tea_logits = self.h3.matmul(self.e3.transpose(-1, -2))
        else:
            stu_logits = self.h2.matmul(self.e2.transpose(-1, -2))
            tea_logits = self.h3.matmul(self.e2.detach().transpose(-1, -2))

        if self.obj == "kl":
            loss = self.cal_kl(stu_logits, tea_logits)
        elif self.obj == "rkl":
            loss = self.cal_rkl(stu_logits, tea_logits)
        elif self.obj == "js":
            loss = self.cal_js(stu_logits, tea_logits)
        elif self.obj == "skl":
            loss = self.cal_skl(stu_logits, tea_logits)
        elif self.obj == "srkl":
            loss = self.cal_srkl(stu_logits, tea_logits)
        elif self.obj == "akl":
            loss = self.cal_akl(stu_logits, tea_logits)
        
        return loss


all_kl_curve = []
all_stu_kl_curve = []

for i in trange(simulation_times):
    
    model = Net(data_num, cls_num, device, obj)
    optim = SGD(model.parameters(), lr=lr, weight_decay=0.0)

    iters = 1000
    min_loss = 100
    kl_curve = []
    for i in range(iters):
        loss = model(False)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        kl_curve.append(loss.data)
        if loss < min_loss:
            plot_h1 = model.h1.data.clone()
            min_loss = loss

    kl_curve = torch.stack(kl_curve, 0)

    min_loss = 100
    stu_kl_curve = []
    for i in range(iters):
        loss = model(True)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        stu_kl_curve.append(loss.data)
        if loss < min_loss:
            plot_h2 = model.h2.data.clone()
            min_loss = loss

    stu_kl_curve = torch.stack(stu_kl_curve, 0)

    all_kl_curve.append(kl_curve)
    all_stu_kl_curve.append(stu_kl_curve)

all_kl_curve = torch.stack(all_kl_curve, 0)
all_stu_kl_curve = torch.stack(all_stu_kl_curve, 0)

mean_kl_curve = all_kl_curve.mean(0).cpu()
mean_stu_kl_curve = all_stu_kl_curve.mean(0).cpu()

std_kl_curve = all_kl_curve.std(0).cpu()
std_stu_kl_curve = all_stu_kl_curve.std(0).cpu()

lim = iters
plt.plot(list(range(lim)), mean_kl_curve[:lim], label="Different Prediction Heads")
plt.fill_between(list(range(lim)), mean_kl_curve - std_kl_curve, mean_kl_curve + std_kl_curve, alpha=0.2)
plt.plot(list(range(lim)), mean_stu_kl_curve[:lim], label="Shared Prediction Head")
plt.fill_between(list(range(lim)), mean_stu_kl_curve - std_stu_kl_curve, mean_stu_kl_curve + std_stu_kl_curve, alpha=0.2)
plt.xlabel("Iterations")
plt.ylabel("KD Loss")
plt.legend()
# plt.savefig(save_path + "loss_curve_average.eps")
# plt.savefig(save_path + "loss_curve_average.pdf")
plt.savefig(save_path + "loss_curve_average.png")