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
    # (1a) 相似度分数: (..., L, d_k) @ (..., d_k, S) -> (..., L, S)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    # (1b) 屏蔽: mask 为 False 的位置填 -inf，softmax 后权重恰好为 0
    #      填 0 不行: e^0 = 1，被屏蔽的位置仍会分到权重
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # (1c) 对每个 query（每一行），在所有 key 上归一化成概率
    weights = torch.softmax(scores, dim=-1)

    # (1d) 加权求和: (..., L, S) @ (..., S, d_v) -> (..., L, d_v)
    out = weights @ v
    return out, weights


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
    # 全 True 的 (L, L) 矩阵 -> tril 只保留下三角（含对角线），其余置为 False
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))


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

        # (3a) 四个线性层，可学习参数全在这里: 4 * (D*D + D)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)  # 把输入变成"提问"
        self.W_k = nn.Linear(d_model, d_model, bias=bias)  # 把输入变成"索引标签"
        self.W_v = nn.Linear(d_model, d_model, bias=bias)  # 把输入变成"要传递的内容"
        self.W_o = nn.Linear(d_model, d_model, bias=bias)  # 把各个头的结果混合起来

    def split_heads(self, x):
        """(B, T, D) -> (B, H, T, Dh)"""
        B, T, _ = x.shape
        # (B, T, D) -> (B, T, H, Dh): 把最后一维 D 切成 H 段，每段 Dh 维（只改形状，不动数据）
        x = x.view(B, T, self.n_heads, self.d_head)
        # (B, T, H, Dh) -> (B, H, T, Dh): 把 H 挪到前面，当作类似 batch 的维度
        return x.transpose(1, 2)

    def merge_heads(self, x):
        """(B, H, T, Dh) -> (B, T, D)，split_heads 的逆操作"""
        B, H, T, Dh = x.shape
        # (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, D)
        # transpose 后内存不连续，view 会报错，所以用 reshape（必要时会自动复制数据）
        return x.transpose(1, 2).reshape(B, T, H * Dh)

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
        # 1. 线性投影                       (B, L, D)  (B, S, D)  (B, S, D)
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # 2. 拆成多头                       (B, H, L, Dh)  (B, H, S, Dh)  (B, H, S, Dh)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        # 3. 所有头并行做 attention         out (B, H, L, Dh), weights (B, H, L, S)
        #    mask (L, S) 会自动广播到 (B, H, L, S)
        out, weights = scaled_dot_product_attention(q, k, v, mask)

        # 4. 合并多头                       (B, L, D)
        out = self.merge_heads(out)

        # 5. 输出投影                       (B, L, D)
        return self.W_o(out), weights
