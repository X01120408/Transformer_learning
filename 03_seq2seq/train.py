"""
训练 Seq2Seq（已写好，不需要修改）

    python train.py --task reverse                     # 倒序（默认自回归解码）
    python train.py --task sort
    python train.py --task add
    python train.py --task add --decoder parallel      # ACT 风格的并行解码
    python train.py --task reverse --steps 500         # 快速试跑

训练过程中会打印 loss 和准确率；结束后会:
    - 打印几个测试样例（✓ / ✗）
    - 保存 cross-attention 热力图 cross_attn_<任务>_<解码方式>.png
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import TASKS, VOCAB, VOCAB_SIZE, make_batch, to_str
from seq2seq import ParallelSeq2Seq, Seq2Seq, Seq2SeqConfig

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="reverse", choices=list(TASKS))
ap.add_argument("--decoder", default="autoregressive", choices=["autoregressive", "parallel"])
ap.add_argument("--n_layer", type=int, default=2)
ap.add_argument("--n_head", type=int, default=4)
ap.add_argument("--d_model", type=int, default=128)
ap.add_argument("--batch_size", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--eval_interval", type=int, default=250)
args = ap.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

cfg = Seq2SeqConfig(vocab_size=VOCAB_SIZE, n_layer=args.n_layer, n_head=args.n_head,
                    d_model=args.d_model, **TASKS[args.task])
Model = Seq2Seq if args.decoder == "autoregressive" else ParallelSeq2Seq
model = Model(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)

print(f"任务 {args.task}，解码方式 {args.decoder}，参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M，设备 {device}")
src, tgt = make_batch(args.task, 1)
print(f"样例: {to_str(src[0])}  ->  {to_str(tgt[0])}\n")


@torch.no_grad()
def evaluate(n=2000):
    """
    用全新随机生成的数据测试（训练时从没见过这些具体样例）。
    注意这里用的是 generate（推理），而不是 forward（训练时的 teacher forcing）。
        字符准确率: 每一位单独算对错
        整体准确率: 整串全对才算对 —— 这才是真正有用的指标
    """
    model.eval()
    src, tgt = make_batch(args.task, n, device)
    pred = model.generate(src)
    model.train()
    return (pred == tgt).float().mean().item(), (pred == tgt).all(1).float().mean().item()


t0 = time.time()
for step in range(args.steps + 1):
    if step % args.eval_interval == 0 or step == args.steps:
        tok_acc, seq_acc = evaluate()
        print(f"[step {step:5d}] loss {loss.item() if step else float('nan'):.4f}   "
              f"字符准确率 {tok_acc:6.1%}   整体准确率 {seq_acc:6.1%}   ({time.time() - t0:.0f}s)")
    if step == args.steps:
        break
    src, tgt = make_batch(args.task, args.batch_size, device)   # 每一步都是全新的数据
    _, loss = model(src, tgt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

# ----------------------------------------------------------------------------- 测试样例
model.eval()
torch.manual_seed(123)
src, tgt = make_batch(args.task, 6, device)
pred = model.generate(src)
print("\n测试样例:")
for s, t, p in zip(src, tgt, pred):
    print(f"  {to_str(s):<20} ->  预测 {to_str(p):<20} {'✓' if torch.equal(t, p) else '✗ 正确答案 ' + to_str(t)}")

# ----------------------------------------------------------------------------- cross-attention 热力图
# 把 decoder 最后一层的 cross-attention 权重画出来：输出的每一位，在看输入的哪些位？
s, p = src[:1], pred[:1]
if args.decoder == "autoregressive":
    model(s, p)            # 用预测结果做一次 teacher forcing 前向，得到每个输出位置的注意力
else:
    model.generate(s)
n_layers = len(model.decoder)
fig, axes = plt.subplots(1, n_layers, figsize=(4.2 * n_layers, 4))
for i, (ax, layer) in enumerate(zip(axes if n_layers > 1 else [axes], model.decoder)):
    w = layer.cross_weights[0].mean(0).cpu()            # 对所有头取平均: (T, S)
    ax.imshow(w, cmap="viridis", vmin=0)
    ax.set_xticks(range(w.shape[1]), [VOCAB[i] for i in s[0].tolist()])
    ax.set_yticks(range(w.shape[0]), [VOCAB[i] for i in p[0].tolist()])
    ax.set_xlabel("input (encoder)")
    ax.set_ylabel("output (decoder)")
    ax.set_title(f"decoder layer {i + 1} cross-attention")
fig.suptitle(f"task = {args.task}, decoder = {args.decoder}")
out = HERE / f"cross_attn_{args.task}_{args.decoder}.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\n已保存 {out.name}（用时 {time.time() - t0:.0f}s）")
