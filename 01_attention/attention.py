"""
项目 1：手写 Attention
=====================

你的任务：把所有标了 TODO 的地方补完，并删掉对应的 `raise NotImplementedError`。

建议顺序：
    Step 1  scaled_dot_product_attention   -> python test_attention.py 1
    Step 2  make_causal_mask               -> python test_attention.py 2
    Step 3  MultiHeadAttention             -> python test_attention.py 3
    全部通过后                               -> python visualize.py

规则：
    - 不许调用 F.scaled_dot_product_attention / nn.MultiheadAttention（测试会拿它们当标准答案对比）
    - 可以用 torch 的基本操作：@ / matmul、transpose、view、reshape、softmax、masked_fill、tril ...
    - 写代码时多 print(x.shape)，写完再删掉
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Step 1: Scaled Dot-Product Attention
# =============================================================================
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V

    参数:
        q:    (..., L, d_k)   L = query 的个数（"谁在提问"）
        k:    (..., S, d_k)   S = key/value 的个数（"可以被查的条目"）
        v:    (..., S, d_v)
        mask: 可选，bool 张量，能广播到 (..., L, S)
              True  = 允许关注
              False = 屏蔽（这个位置的注意力权重必须为 0）
              （和 PyTorch 的 F.scaled_dot_product_attention 约定一致）

        "..." 表示前面可以有任意多个维度，比如 (B, L, d) 或 (B, H, L, d)。
        只要你只对最后两维做操作，代码就自动支持这些情况。

    返回:
        out:     (..., L, d_v)
        weights: (..., L, S)   注意力权重，每一行之和为 1
    """
    # TODO(1a): 计算注意力分数 scores = Q @ K^T / sqrt(d_k)，形状 (..., L, S)
    #   提示: k.transpose(-2, -1) 交换最后两维；d_k = q.size(-1)

    # TODO(1b): 如果 mask 不是 None，把 mask 为 False 的位置填成 -inf
    #   提示: scores.masked_fill(条件, 值) —— 条件为 True 的位置会被填
    #   想一想: 为什么填 -inf 而不是 0？

    # TODO(1c): 在最后一维 (dim=-1) 上做 softmax，得到 weights

    # TODO(1d): out = weights @ v

    raise NotImplementedError("Step 1 还没实现")


# =============================================================================
# Step 2: Causal Mask（因果掩码）
# =============================================================================
def make_causal_mask(L, device=None):
    """
    生成 (L, L) 的 bool 下三角矩阵（含对角线），例如 L=4:
        [[T, F, F, F],
         [T, T, F, F],
         [T, T, T, F],
         [T, T, T, T]]
    第 i 行表示：第 i 个位置只能看到第 0..i 个位置，看不到"未来"。
    """
    # TODO(2): 一行就够了。提示: torch.ones(..., dtype=torch.bool) 和 torch.tril

    raise NotImplementedError("Step 2 还没实现")


# =============================================================================
# Step 3: Multi-Head Attention
# =============================================================================
class MultiHeadAttention(nn.Module):
    """
    把 d_model 维切成 n_heads 份，每个头在自己的 d_head 维子空间里独立做 attention，
    最后拼回来再过一个线性层。

    形状流程（B=batch, L=query 长度, S=key 长度, H=n_heads, D=d_model, Dh=d_head）:
        query (B, L, D) --W_q--> (B, L, D) --split--> (B, H, L, Dh) ┐
        key   (B, S, D) --W_k--> (B, S, D) --split--> (B, H, S, Dh) ├─ attention ─> (B, H, L, Dh)
        value (B, S, D) --W_v--> (B, S, D) --split--> (B, H, S, Dh) ┘
                                                     --merge--> (B, L, D) --W_o--> (B, L, D)
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # TODO(3a): 定义 4 个线性层，都是 nn.Linear(d_model, d_model, bias=bias)
        #   名字必须是 self.W_q, self.W_k, self.W_v, self.W_o（测试要用这些名字拷贝权重）

        raise NotImplementedError("Step 3a 还没实现（__init__）")

    def split_heads(self, x):
        """(B, T, D) -> (B, H, T, Dh)"""
        # TODO(3b): 先 view 成 (B, T, H, Dh)，再 transpose 成 (B, H, T, Dh)
        #   提示: B, T, _ = x.shape

        raise NotImplementedError("Step 3b 还没实现（split_heads）")

    def merge_heads(self, x):
        """(B, H, T, Dh) -> (B, T, D)，split_heads 的逆操作"""
        # TODO(3c): 先 transpose 回 (B, T, H, Dh)，再变成 (B, T, D)
        #   注意: transpose 之后内存不连续，直接 .view 会报错。用 .reshape，或者 .contiguous().view
        #   想一想: 为什么不能直接 x.view(B, T, D)，跳过 transpose？

        raise NotImplementedError("Step 3c 还没实现（merge_heads）")

    def forward(self, query, key, value, mask=None):
        """
        参数:
            query: (B, L, D)
            key:   (B, S, D)
            value: (B, S, D)
              - self-attention:  query = key = value = x
              - cross-attention: query 来自一处，key = value 来自另一处（ACT 的 decoder 就是这样）
            mask: 可选 bool，形状 (L, S) 或能广播到 (B, H, L, S)，True = 允许

        返回:
            out:     (B, L, D)
            weights: (B, H, L, S)   每个头各自的注意力权重
        """
        # TODO(3d): 五步走
        #   1. 线性投影:  q = self.W_q(query)，k、v 同理
        #   2. 拆成多头:  split_heads
        #   3. 做 attention: 直接调用你在 Step 1 写的函数（它支持 "..." 维度，多头不用写循环）
        #   4. 合并多头:  merge_heads
        #   5. 输出投影:  self.W_o

        raise NotImplementedError("Step 3d 还没实现（forward）")
