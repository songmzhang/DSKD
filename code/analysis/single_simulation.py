import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from tqdm import trange
import torch.nn as nn
from torch.optim import SGD
import os


data_num = 100
cls_num = 10000
lr = 1.0
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

    def cal_mse(self, a, b):
        mse = F.mse_loss(a, b, reduction="mean")
        return mse

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

for i in trange(1):
    
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
    for i in range(iters):
        loss = model(True)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        if loss < min_loss:
            plot_h2 = model.h2.data.clone()
            min_loss = loss

yhigh = int(max(model.h0[:, 1].max().item(), plot_h1[:, 1].max().item(), plot_h2[:, 1].max().item(), model.h3[:, 1].max().item())) + 1.5
ylow = int(min(model.h0[:, 1].min().item(), plot_h1[:, 1].min().item(), plot_h2[:, 1].min().item(), model.h3[:, 1].min().item())) - 1.5
xhigh = int(max(model.h0[:, 0].max().item(), plot_h1[:, 0].max().item(), plot_h2[:, 0].max().item(), model.h3[:, 0].max().item())) + 1.5
xlow = int(min(model.h0[:, 0].min().item(), plot_h1[:, 0].min().item(), plot_h2[:, 0].min().item(), model.h3[:, 0].min().item())) - 1.5

plt.xlim(xlow, xhigh)
plt.ylim(ylow, yhigh)
plt.scatter(model.h0[:, 0].detach().cpu().numpy(), model.h0[:, 1].detach().cpu().numpy(), marker="^", c="r", label="Student")
plt.scatter(model.h3[:, 0].detach().cpu().numpy(), model.h3[:, 1].detach().cpu().numpy(), marker="*", c="b", label="Teacher")
plt.legend()
# plt.savefig(save_path + "before.eps")
# plt.savefig(save_path + "before.pdf")
plt.savefig(save_path + "before.png")

plt.cla()
plt.xlim(xlow, xhigh)
plt.ylim(ylow, yhigh)
plt.scatter(plot_h1[:, 0].detach().cpu().numpy(), plot_h1[:, 1].detach().cpu().numpy(), marker="^", c="r", label="Student")
plt.scatter(model.h3[:, 0].detach().cpu().numpy(), model.h3[:, 1].detach().cpu().numpy(), marker="*", c="b", label="Teacher")
plt.legend()
# plt.savefig(save_path + "after_diff_head.eps")
# plt.savefig(save_path + "after_diff_head.pdf")
plt.savefig(save_path + "after_diff_head.png")

plt.cla()
plt.xlim(xlow, xhigh)
plt.ylim(ylow, yhigh)
plt.scatter(plot_h2[:, 0].detach().cpu().numpy(), plot_h2[:, 1].detach().cpu().numpy(), marker="^", c="r", label="Student")
plt.scatter(model.h3[:, 0].detach().cpu().numpy(), model.h3[:, 1].detach().cpu().numpy(), marker="*", c="b", label="Teacher")
plt.legend()
# plt.savefig(save_path + "after_share_head.eps")
# plt.savefig(save_path + "after_share_head.pdf")
plt.savefig(save_path + "after_share_head.png")
