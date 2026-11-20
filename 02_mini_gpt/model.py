"""
项目 2：字符级 mini-GPT —— 模型部分
==================================

你的任务：补完所有 TODO，并删掉对应的 `raise NotImplementedError`。

建议顺序：
    Step 1  FeedForward        -> python test_model.py 1
    Step 2  Block              -> python test_model.py 2
    Step 3  GPT 的 __init__ / forward -> python test_model.py 3
    Step 4  GPT.generate       -> python test_model.py 4
    全部通过后                   -> python train.py

本文件直接复用项目 1 你写的 MultiHeadAttention 和 make_causal_mask。
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 把项目 1 的文件夹加入搜索路径，这样就能 import 你写的 attention.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_attention"))
from attention import MultiHeadAttention, make_causal_mask  # noqa: E402


@dataclass
class GPTConfig:
    vocab_size: int          # 词表大小（字符级 = 文本里有多少种不同的字符）
    block_size: int = 128    # 上下文长度：模型一次最多看多少个字符
    n_layer: int = 4         # Transformer Block 的层数
    n_head: int = 4          # 注意力头数
    d_model: int = 256       # 每个 token 的向量维度
    dropout: float = 0.1


# =============================================================================
# Step 1: FeedForward（前馈网络）
# =============================================================================
class FeedForward(nn.Module):
    """
    对每个位置**单独**做的两层 MLP：  D -> 4D -> GELU -> D -> Dropout
    Attention 负责"在位置之间交换信息"，FeedForward 负责"在每个位置上加工信息"。
    """

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        # (1a) D -> 4D -> GELU -> D -> Dropout
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # 升维: 给"加工"留出更大的空间
            nn.GELU(),                         # 非线性: 没有它，两个 Linear 叠起来仍然只是一个 Linear
            nn.Linear(4 * d_model, d_model),   # 降维: 回到 D，才能和残差相加
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # (1b) Linear 只作用于最后一维，所以每个位置是独立计算的
        return self.net(x)


# =============================================================================
# Step 2: Transformer Block
# =============================================================================
class Block(nn.Module):
    """
    一个 Transformer Block（Pre-LN 结构，GPT-2 及之后的主流写法）:

        x ──┬──> LayerNorm ──> 因果 Self-Attention ──> Dropout ──(+)──┬──> LayerNorm ──> FeedForward ──(+)──> 输出
            └──────────────────── 残差连接 ─────────────────────────┘  └────────────── 残差连接 ────────┘

    即:
        x = x + Dropout(Attention(LN1(x)))
        x = x + FeedForward(LN2(x))
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # (2a) 参数量: Attention 4D²+4D, FeedForward 8D²+5D, 两个 LayerNorm 各 2D
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_head)   # 项目 1 你写的
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.dropout)          # Step 1
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D)"""
        T = x.size(1)
        # (T, T) 下三角，会自动广播到 (B, H, T, T)；device 要和 x 一致，否则 GPU 上会报错
        mask = make_causal_mask(T, device=x.device)

        # 子层 1: 先 LayerNorm，再因果自注意力（q = k = v = h），结果加回 x
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, mask)   # 返回 (out, weights)，这里只要 out
        x = x + self.drop(a)

        # 子层 2: 先 LayerNorm，再前馈网络，结果加回 x
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# Step 3 & 4: GPT
# =============================================================================
class GPT(nn.Module):
    """
    idx (B, T) 字符编号
      │ tok_emb: 每个编号查表得到一个 D 维向量          (B, T, D)
      │ + pos_emb: 每个位置 0..T-1 也查表得到一个向量   (T, D) 广播相加
      │ dropout
      │ n_layer 个 Block
      │ ln_f: 最后一个 LayerNorm
      │ head: Linear(D -> vocab_size)
      ▼
    logits (B, T, vocab_size)   第 t 个位置的输出 = 对"第 t+1 个字符是什么"的打分
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # 字符编号 -> 向量
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)   # 位置编号 -> 向量（可学习的位置编码）
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)                       # Pre-LN 结构最后要再归一化一次
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # 向量 -> 每个字符的打分

        self.apply(self._init_weights)  # 已写好: 按 GPT-2 的方式初始化权重

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        参数:
            idx:     (B, T) long，字符编号，T <= block_size
            targets: (B, T) long，可选。targets[:, t] 是 idx[:, t] 的"下一个字符"
        返回:
            logits: (B, T, vocab_size)
            loss:   标量交叉熵；targets 为 None 时返回 None
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"序列长度 {T} 超过了 block_size {self.cfg.block_size}"

        # 1-3. 字符向量 + 位置向量:  (B, T, D) + (T, D) -> (B, T, D)
        pos = torch.arange(T, device=idx.device)   # [0, 1, ..., T-1]
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        # 4. 逐层通过 Transformer Block，形状始终是 (B, T, D)
        for block in self.blocks:
            x = block(x)

        # 5. 最后的 LayerNorm + 输出层:  (B, T, D) -> (B, T, vocab_size)
        logits = self.head(self.ln_f(x))

        # 6. 交叉熵要求 (N, C) 和 (N,)，所以把 B、T 两维拍平成 B*T 个样本
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成：每次预测一个字符，拼到末尾，再预测下一个……

        参数:
            idx:            (B, T0) 开头的字符编号（prompt）
            max_new_tokens: 要生成多少个新字符
            temperature:    温度。<1 更保守（更像 argmax），>1 更随机
            top_k:          只在概率最高的 k 个字符里采样；None 表示不限制
        返回:
            (B, T0 + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 1. 只保留最后 block_size 个字符（位置编码只有 block_size 个）
            idx_cond = idx[:, -self.cfg.block_size:]
            # 2. 前向，得到每个位置的打分 (B, T, vocab_size)
            logits, _ = self(idx_cond)
            # 3. 只有最后一个位置的输出才是"下一个字符"的预测；除以温度调整分布的尖锐程度
            logits = logits[:, -1, :] / temperature
            # 4. top-k: 第 k 名之后的字符全部设为 -inf，softmax 后概率为 0
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            # 5-6. 变成概率，按概率随机抽一个（而不是取最大值）
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)
            # 7. 拼到末尾，进入下一轮
            idx = torch.cat([idx, next_id], dim=1)
        return idx
