"""
Step 1 练习脚本：用很小的数字，一步一步演示 attention 用到的每个 torch 操作

    python step1_playground.py

建议: 先运行一遍看输出，再逐段读代码，最后在 VSCode 里改改数字重新运行。
"""
import math

import torch

torch.set_printoptions(precision=3, sci_mode=False)


def title(s):
    print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


# -----------------------------------------------------------------------------
title("准备数据: 2 个 query, 3 个 key/value, 每个向量 2 维 (d_k = 2)")
# -----------------------------------------------------------------------------
q = torch.tensor([[1.0, 0.0],    # query 0: 只关心第 1 个特征
                  [0.0, 1.0]])   # query 1: 只关心第 2 个特征
k = torch.tensor([[1.0, 0.0],    # key 0: 有第 1 个特征
                  [0.0, 1.0],    # key 1: 有第 2 个特征
                  [1.0, 1.0]])   # key 2: 两个特征都有
v = torch.tensor([[10.0, 0.0],   # value 0
                  [0.0, 10.0],   # value 1
                  [5.0, 5.0]])   # value 2
print("q.shape =", tuple(q.shape), " -> (L=2, d_k=2)")
print("k.shape =", tuple(k.shape), " -> (S=3, d_k=2)")
print("v.shape =", tuple(v.shape), " -> (S=3, d_v=2)")


# -----------------------------------------------------------------------------
title("① 转置: k.transpose(-2, -1) 交换最后两个维度")
# -----------------------------------------------------------------------------
kt = k.transpose(-2, -1)
print("k  =\n", k, "  shape", tuple(k.shape))
print("kᵀ =\n", kt, "  shape", tuple(kt.shape))
print("-2 和 -1 表示'倒数第二维'和'最后一维'。用负数的好处: 前面加了 batch 维也不用改代码")


# -----------------------------------------------------------------------------
title("② 矩阵乘法: q @ kᵀ  得到相似度分数")
# -----------------------------------------------------------------------------
scores = q @ kt
print("scores = q @ kᵀ =\n", scores, "  shape", tuple(scores.shape), "-> (L, S)")
print("第 i 行第 j 列 = query i 和 key j 的点积。手算验证一个:")
print(f"  scores[0, 2] = q[0]·k[2] = 1*1 + 0*1 = {scores[0, 2].item()}")


# -----------------------------------------------------------------------------
title("③ 缩放: 除以 sqrt(d_k)")
# -----------------------------------------------------------------------------
d_k = q.size(-1)  # 最后一维的大小 = 2
scores = scores / math.sqrt(d_k)
print(f"d_k = q.size(-1) = {d_k},  sqrt(d_k) = {math.sqrt(d_k):.3f}")
print("缩放后 scores =\n", scores)


# -----------------------------------------------------------------------------
title("④ mask: 把不允许看的位置填成 -inf")
# -----------------------------------------------------------------------------
mask = torch.tensor([[True, True, False],   # query 0 不允许看 key 2
                     [True, True, True]])   # query 1 全都能看
print("mask  (True = 允许) =\n", mask)
print("~mask (取反, True = 要屏蔽) =\n", ~mask)
masked = scores.masked_fill(~mask, float("-inf"))
print("scores.masked_fill(~mask, -inf) =\n", masked)
print("注意: masked_fill 不会修改原张量，而是返回一个新张量，要用变量接住它！")
print("      scores.masked_fill(...)            ← 错，结果被丢掉了")
print("      scores = scores.masked_fill(...)   ← 对")


# -----------------------------------------------------------------------------
title("⑤ softmax: 把每一行分数变成概率")
# -----------------------------------------------------------------------------
weights = torch.softmax(masked, dim=-1)  # 也可以写成 masked.softmax(dim=-1)
print("softmax(dim=-1) =\n", weights)
print("每行之和:", weights.sum(dim=-1), " ← 都是 1")
print("被 -inf 屏蔽的位置权重恰好是 0（因为 e^-inf = 0）")
print("\n对比: 如果错用 dim=0（按列归一化）:\n", torch.softmax(masked, dim=0))
print("这时变成了每一列之和为 1，含义完全不对")
print("\n对比: 如果 mask 填 0 而不是 -inf:\n", torch.softmax(scores.masked_fill(~mask, 0.0), dim=-1))
print("屏蔽位置的权重不是 0，信息泄露了！")


# -----------------------------------------------------------------------------
title("⑥ 加权求和: weights @ v")
# -----------------------------------------------------------------------------
out = weights @ v
print("out = weights @ v =\n", out, "  shape", tuple(out.shape), "-> (L, d_v)")
manual = weights[1, 0] * v[0] + weights[1, 1] * v[1] + weights[1, 2] * v[2]
print("手算 out[1] = w[1,0]*v[0] + w[1,1]*v[1] + w[1,2]*v[2] =", manual)
print("解读: query 1 只关心第 2 个特征，所以它从 key 1 和 key 2 取得最多信息，")
print("      输出的第 2 维比第 1 维大")


# -----------------------------------------------------------------------------
title("⑦ 同样的代码，前面多了 batch 和 head 维度也照样能用")
# -----------------------------------------------------------------------------
B, H, L, S, d = 2, 4, 5, 7, 8
qb, kb, vb = torch.randn(B, H, L, d), torch.randn(B, H, S, d), torch.randn(B, H, S, d)
sb = qb @ kb.transpose(-2, -1)
print(f"q {tuple(qb.shape)} @ kᵀ {tuple(kb.transpose(-2, -1).shape)} = {tuple(sb.shape)}")
print("@ 只对最后两维做矩阵乘法，前面的维度 (B, H) 原样保留，相当于自动做了 B*H 次")
print("如果写成 k.transpose(0, 1) 或 k.T，遇到这种 4 维输入就错了")


print("\n" + "=" * 60)
print("现在回到 attention.py，把 ①~⑥ 组合成 scaled_dot_product_attention 函数。")
print("提示: mask 是可选参数，只有 mask is not None 时才做第 ④ 步。")
print("写完运行: python test_attention.py 1")
