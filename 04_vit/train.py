"""
在 CIFAR-10 上训练 ViT（已写好，不需要修改；建议通读）

    python train.py                     # 默认 ViT，30 个 epoch
    python train.py --epochs 5          # 快速试跑
    python train.py --model cnn         # 对照组: 一个参数更少的小 CNN
    python train.py --no_aug            # 关闭数据增强（实验用）

训练结束后生成:
    checkpoints/<model>.pt   模型权重（visualize.py 要用）
    curve_<model>.png        训练曲线
"""
import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit import ViT, ViTConfig

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="vit", choices=["vit", "cnn"])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch_size", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--weight_decay", type=float, default=0.05)
ap.add_argument("--patch_size", type=int, default=4)
ap.add_argument("--d_model", type=int, default=192)
ap.add_argument("--n_layer", type=int, default=6)
ap.add_argument("--n_head", type=int, default=3)
ap.add_argument("--no_aug", action="store_true", help="关闭数据增强")
args = ap.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# =============================================================================
# 1. 数据: 整个 CIFAR-10 只有 180MB，直接全部放进显存，数据增强也在 GPU 上做（比 DataLoader 快得多）
# =============================================================================
npz = HERE / "data" / "cifar10.npz"
if not npz.exists():
    raise SystemExit("找不到数据，先运行: python prepare_data.py")
d = np.load(npz)
x_train = torch.from_numpy(d["x_train"]).to(device)   # (50000, 3, 32, 32) uint8
y_train = torch.from_numpy(d["y_train"]).to(device)
x_test = torch.from_numpy(d["x_test"]).to(device)
y_test = torch.from_numpy(d["y_test"]).to(device)

MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)


def normalize(x_uint8):
    """uint8 [0, 255] -> float，每个通道减均值、除标准差"""
    return (x_uint8.float() / 255.0 - MEAN) / STD


def augment(x):
    """
    数据增强: 每张图随机平移（四周补 4 像素后随机裁回 32×32）+ 50% 概率水平翻转。
    让模型每个 epoch 看到的图都略有不同，相当于扩充了数据量，缓解过拟合。
    """
    B = x.size(0)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")                        # (B, 3, 40, 40)
    dy, dx = torch.randint(0, 9, (B,), device=x.device), torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (dy[:, None] + ar)[:, None, :, None]                         # (B, 1, 32, 1)
    cols = (dx[:, None] + ar)[:, None, None, :]                         # (B, 1, 1, 32)
    x = xp[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


# =============================================================================
# 2. 模型
# =============================================================================
class SmallCNN(nn.Module):
    """对照组: 一个普通的小 CNN（参数量不到默认 ViT 的一半）"""

    def __init__(self, c=128):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())

        self.net = nn.Sequential(block(3, c // 2), nn.MaxPool2d(2),        # 32 -> 16
                                 block(c // 2, c), nn.MaxPool2d(2),        # 16 -> 8
                                 block(c, 2 * c), nn.AdaptiveAvgPool2d(1),  # 8 -> 1
                                 nn.Flatten(), nn.Linear(2 * c, 10))

    def forward(self, x):
        return self.net(x)


if args.model == "vit":
    cfg = ViTConfig(patch_size=args.patch_size, d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head)
    model = ViT(cfg).to(device)
    print(f"ViT: patch {cfg.patch_size}×{cfg.patch_size} -> {cfg.n_patches} 个 patch + 1 个 CLS，"
          f"d_model {cfg.d_model}，{cfg.n_layer} 层，{cfg.n_head} 头")
else:
    cfg = None
    model = SmallCNN().to(device)
print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M，设备 {device}，数据增强 {'关' if args.no_aug else '开'}")

# ViT 常用配方: AdamW + 较大的 weight decay + 预热 + 余弦退火 + label smoothing
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
steps_per_epoch = len(x_train) // args.batch_size
total = args.epochs * steps_per_epoch
warmup = min(5 * steps_per_epoch, total // 10)


def lr_at(step):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def evaluate(x, y, n=None):
    model.eval()
    correct = 0
    for i in range(0, len(x) if n is None else n, 1000):
        with torch.autocast(device, dtype=torch.bfloat16):
            correct += (model(normalize(x[i:i + 1000])).argmax(-1) == y[i:i + 1000]).sum().item()
    model.train()
    return correct / (len(x) if n is None else n)


# =============================================================================
# 3. 训练
# =============================================================================
hist = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
step, t0 = 0, time.time()
for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(x_train), device=device)
    losses = []
    for i in range(steps_per_epoch):
        idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
        xb, yb = x_train[idx], y_train[idx]
        if not args.no_aug:
            xb = augment(xb)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        # bfloat16 混合精度: 3060 上速度约快一倍，精度几乎不受影响
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = F.cross_entropy(model(normalize(xb)), yb, label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        step += 1

    train_acc = evaluate(x_train, y_train, n=10000)   # 训练集上抽 1 万张（不做增强）
    test_acc = evaluate(x_test, y_test)
    hist["epoch"].append(epoch)
    hist["train_loss"].append(sum(losses) / len(losses))
    hist["train_acc"].append(train_acc)
    hist["test_acc"].append(test_acc)
    print(f"[epoch {epoch:3d}] loss {hist['train_loss'][-1]:.3f}   训练集准确率 {train_acc:6.1%}   "
          f"测试集准确率 {test_acc:6.1%}   ({time.time() - t0:.0f}s)")

# =============================================================================
# 4. 保存
# =============================================================================
(HERE / "checkpoints").mkdir(exist_ok=True)
torch.save({"model": model.state_dict(), "config": cfg.__dict__ if cfg else None, "hist": hist},
           HERE / "checkpoints" / f"{args.model}.pt")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["epoch"], hist["train_loss"], "o-")
ax[0].set(xlabel="epoch", ylabel="train loss", title="loss")
ax[1].plot(hist["epoch"], hist["train_acc"], "o-", label="train")
ax[1].plot(hist["epoch"], hist["test_acc"], "o-", label="test")
ax[1].set(xlabel="epoch", ylabel="accuracy", title="accuracy")
ax[1].legend()
for a in ax:
    a.grid(alpha=0.3)
fig.suptitle(f"{args.model} on CIFAR-10")
plt.savefig(HERE / f"curve_{args.model}.png", dpi=120, bbox_inches="tight")
print(f"\n最终测试集准确率 {hist['test_acc'][-1]:.1%}，用时 {time.time() - t0:.0f}s")
print(f"已保存 checkpoints/{args.model}.pt 和 curve_{args.model}.png")
if args.model == "vit":
    print("下一步: python visualize.py")
