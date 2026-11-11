"""
用你自己写的 attention 画注意力热力图（全部测试通过后再运行）

    python visualize.py      ->  生成 attention_vis.png

这里用的是最"原始"的 self-attention: Q = K = V = 词向量，没有任何可学习参数。
每个词用一个随机向量表示，相同的词向量相同。
观察: 谁在关注谁？加了因果 mask 之后有什么变化？
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from attention import make_causal_mask, scaled_dot_product_attention

tokens = "the robot picks the cube then the robot places the cube".split()
T, d = len(tokens), 32

torch.manual_seed(0)
vocab = sorted(set(tokens))
emb = {w: torch.randn(d) for w in vocab}
x = torch.stack([emb[w] for w in tokens])  # (T, d)

_, w_full = scaled_dot_product_attention(x, x, x)
_, w_causal = scaled_dot_product_attention(x, x, x, make_causal_mask(T))

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
for ax, w, title in [(axes[0], w_full, "Self-attention (no mask)"),
                     (axes[1], w_causal, "Self-attention (causal mask)")]:
    im = ax.imshow(w.numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(T), tokens, rotation=60, ha="right")
    ax.set_yticks(range(T), tokens)
    ax.set_xlabel("key  (being attended to)")
    ax.set_ylabel("query  (who is asking)")
    ax.set_title(title)
    for i in range(T):
        for j in range(T):
            if w[i, j] > 0.05:
                ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if w[i, j] > 0.5 else "white")
fig.colorbar(im, ax=axes, shrink=0.8)
plt.savefig("attention_vis.png", dpi=120, bbox_inches="tight")
print("已保存 attention_vis.png，在 VSCode 里点开看看")
print("问题 1: 左图中，第二个 'robot' 主要在关注哪些词？为什么？")
print("问题 2: 右图中，第一行为什么只有一个格子是 1.00？")
print("问题 3: 右图中，第一个 'cube' 和第二个 'cube' 的那一行有什么不同？")
