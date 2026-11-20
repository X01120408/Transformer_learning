"""
训练字符级 mini-GPT（已写好，不需要修改；建议通读一遍，每段都有注释）

    python train.py                 # 正式训练，RTX 3060 上约几分钟
    python train.py --overfit       # 先做"单 batch 过拟合"检查: loss 应该很快降到接近 0
    python train.py --max_iters 500 # 快速试跑

训练结束后会生成:
    checkpoints/ckpt.pt   模型权重（给 sample.py 用）
    loss_curve.png        训练 / 验证 loss 曲线
"""
import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(HERE / "data" / "input.txt"))
ap.add_argument("--block_size", type=int, default=128)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--d_model", type=int, default=256)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--max_iters", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--eval_interval", type=int, default=250)
ap.add_argument("--overfit", action="store_true", help="只在一个固定 batch 上训练，检查代码有没有 bug")
args = ap.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")  # 3060 (Ampere) 上启用 TF32，矩阵乘法更快

# =============================================================================
# 1. 数据: 文本 -> 字符编号
# =============================================================================
if not Path(args.data).exists():
    raise SystemExit("找不到数据，先运行: python prepare_data.py")
text = Path(args.data).read_text(encoding="utf-8")

chars = sorted(set(text))                      # 词表: 文本中出现过的所有字符
stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 编号
itos = {i: ch for ch, i in stoi.items()}       # 编号 -> 字符
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print(f"词表大小 {len(chars)}，例: encode('hello') = {encode('hello')}")

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]      # 前 90% 训练，后 10% 验证


def get_batch(split):
    """
    随机截取 batch_size 段长度为 block_size 的文本。
    y 是 x 向右错开一位: 在每个位置上，目标都是"下一个字符"。

        文本:  F i r s t   C i t
        x:     F i r s t   C i
        y:       i r s t   C i t
    一段长 T 的文本，一次就提供了 T 个训练样本。
    """
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([d[i: i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1: i + 1 + args.block_size] for i in ix])
    return x.to(device), y.to(device)


# =============================================================================
# 2. 模型和优化器
# =============================================================================
cfg = GPTConfig(vocab_size=len(chars), block_size=args.block_size, n_layer=args.n_layer,
                n_head=args.n_head, d_model=args.d_model,
                dropout=0.0 if args.overfit else args.dropout)
model = GPT(cfg).to(device)
print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M，设备 {device}")
print(f"初始 loss 理论值 ln({len(chars)}) = {math.log(len(chars)):.3f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)


def lr_at(it):
    """学习率: 前 100 步线性预热，之后余弦下降到 1/10"""
    warmup = 100
    if it < warmup:
        return args.lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, args.max_iters - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(iters=50):
    """在训练集和验证集上各取 iters 个 batch 求平均 loss（单个 batch 的 loss 噪声太大）"""
    model.eval()  # 关闭 dropout
    out = {}
    for split in ["train", "val"]:
        losses = [model(*get_batch(split))[1].item() for _ in range(iters)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def sample(n=200, prompt="\n"):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    s = decode(model.generate(idx, n, temperature=0.8)[0].tolist())
    model.train()
    return s


# =============================================================================
# 3. 训练循环
# =============================================================================
if args.overfit:
    fixed = get_batch("train")
    print("\n[单 batch 过拟合检查] loss 应在几百步内降到 0.1 以下")

history = {"it": [], "train": [], "val": []}
t0 = time.time()
for it in range(args.max_iters + 1):
    # 定期评估 + 打印生成样例，观察模型从"乱码"到"像样"的过程
    if not args.overfit and (it % args.eval_interval == 0 or it == args.max_iters):
        losses = estimate_loss()
        history["it"].append(it)
        history["train"].append(losses["train"])
        history["val"].append(losses["val"])
        print(f"\n[step {it:5d}] train loss {losses['train']:.3f}  val loss {losses['val']:.3f}"
              f"  ({time.time() - t0:.0f}s)")
        print("-" * 50 + "\n" + sample(150).strip() + "\n" + "-" * 50)
    if it == args.max_iters:
        break

    for g in optimizer.param_groups:
        g["lr"] = lr_at(it)

    x, y = fixed if args.overfit else get_batch("train")
    _, loss = model(x, y)                                     # 前向
    optimizer.zero_grad(set_to_none=True)                     # 清空上一步的梯度
    loss.backward()                                           # 反向传播
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪，防止偶发的梯度爆炸
    optimizer.step()                                          # 更新参数

    if args.overfit and it % 50 == 0:
        print(f"step {it:4d}  loss {loss.item():.4f}")

if args.overfit:
    print(f"最终 loss {loss.item():.4f}  ->", "✅ 通过" if loss.item() < 0.1 else "❌ 没降下去，模型可能有 bug")
    raise SystemExit

# =============================================================================
# 4. 保存
# =============================================================================
(HERE / "checkpoints").mkdir(exist_ok=True)
torch.save({"model": model.state_dict(), "config": cfg.__dict__, "chars": chars}, HERE / "checkpoints" / "ckpt.pt")

plt.figure(figsize=(7, 4))
plt.plot(history["it"], history["train"], "o-", label="train")
plt.plot(history["it"], history["val"], "o-", label="val")
plt.axhline(math.log(len(chars)), ls="--", c="gray", label=f"random guess ln({len(chars)})")
plt.xlabel("step")
plt.ylabel("cross-entropy loss")
plt.title("mini-GPT on tiny Shakespeare")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(HERE / "loss_curve.png", dpi=120, bbox_inches="tight")
print(f"\n训练完成，用时 {time.time() - t0:.0f}s")
print("已保存 checkpoints/ckpt.pt 和 loss_curve.png")
print("下一步: python sample.py --prompt 'ROMEO:'")
