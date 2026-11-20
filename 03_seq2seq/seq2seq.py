"""
项目 3：Encoder-Decoder Transformer（Seq2Seq）
=============================================

你的任务：补完所有 TODO，并删掉对应的 `raise NotImplementedError`。

建议顺序：
    Step 1  EncoderLayer                -> python test_seq2seq.py 1
    Step 2  DecoderLayer                -> python test_seq2seq.py 2
    Step 3  Seq2Seq 的 encode / decode / forward -> python test_seq2seq.py 3
    Step 4  Seq2Seq.generate（自回归）    -> python test_seq2seq.py 4
    Step 5  ParallelSeq2Seq（ACT 风格）   -> python test_seq2seq.py 5
    全部通过后                            -> python train.py

复用的模块:
    项目 1: MultiHeadAttention, make_causal_mask
    项目 2: FeedForward
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_attention"))
sys.path.insert(0, str(ROOT / "02_mini_gpt"))
from attention import MultiHeadAttention, make_causal_mask  # noqa: E402
from model import FeedForward  # noqa: E402  （项目 2 的 model.py）

from data import BOS  # noqa: E402


@dataclass
class Seq2SeqConfig:
    vocab_size: int
    src_len: int             # 输入序列长度
    tgt_len: int             # 输出序列长度
    n_layer: int = 2         # encoder 和 decoder 各有几层
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.0


# =============================================================================
# Step 1: Encoder Layer
# =============================================================================
class EncoderLayer(nn.Module):
    """
    和项目 2 的 Block 几乎一样，唯一区别: **没有因果掩码**。
    Encoder 的任务是"理解输入"，输入是一次性全部给定的，每个位置都可以看到所有位置（双向）。

        x = x + Dropout(SelfAttention(LN1(x)))      ← 不传 mask
        x = x + FeedForward(LN2(x))
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        # (1a) 和项目 2 的 Block 完全一样
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_head)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """x: (B, S, D) -> (B, S, D)"""
        # (1b) 唯一区别: 不传 mask —— 每个位置都能看到整个输入（双向）
        h = self.ln1(x)
        a, _ = self.attn(h, h, h)
        x = x + self.drop(a)
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# Step 2: Decoder Layer
# =============================================================================
class DecoderLayer(nn.Module):
    """
    比 Encoder 多了一个子层 —— Cross-Attention:

        y = y + Dropout(SelfAttention(LN1(y), mask=因果掩码 或 None))   ① 输出序列内部交流
        y = y + Dropout(CrossAttention(q=LN2(y), k=v=memory))            ② 去 encoder 的输出里查信息
        y = y + FeedForward(LN3(y))                                      ③ 加工

    memory 是 encoder 的输出 (B, S, D)。cross-attention 时:
        query 来自 decoder 自己 (B, T, D)，key/value 来自 memory (B, S, D)，T 和 S 可以不同。

    参数 causal:
        True  —— 标准 Transformer / GPT 式的自回归解码，不能看后面的位置
        False —— ACT 式的并行解码，所有查询之间可以互相看（Step 5 会用到）
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.n_head
        self.ln1 = nn.LayerNorm(D)
        self.self_attn = MultiHeadAttention(D, H)    # ① 输出序列内部交流
        self.ln2 = nn.LayerNorm(D)
        self.cross_attn = MultiHeadAttention(D, H)   # ② 去 encoder 的输出里查信息（和 ① 不共享参数）
        self.ln3 = nn.LayerNorm(D)
        self.ffn = FeedForward(D, cfg.dropout)       # ③ 加工
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, y, memory, causal=True):
        """
        y:      (B, T, D)  decoder 这一侧的序列
        memory: (B, S, D)  encoder 的输出
        返回:   (B, T, D)
        """
        # 自回归时用因果掩码；ACT 式并行解码时不用
        mask = make_causal_mask(y.size(1), device=y.device) if causal else None

        # ① self-attention: q = k = v 都来自 decoder 自己
        h = self.ln1(y)
        a, _ = self.self_attn(h, h, h, mask)
        y = y + self.drop(a)

        # ② cross-attention: q 来自 decoder (B,T,D)，k = v 来自 encoder 的 memory (B,S,D)
        #    权重 w 的形状 (B, H, T, S): 输出第 t 位对输入第 s 位的关注程度
        h = self.ln2(y)
        c, w = self.cross_attn(h, memory, memory)
        self.cross_weights = w.detach()   # 保存下来画热力图；detach 表示不参与反向传播
        y = y + self.drop(c)

        # ③ 前馈
        y = y + self.ffn(self.ln3(y))
        return y


# =============================================================================
# Step 3 & 4: Seq2Seq（标准 Transformer，自回归解码）
# =============================================================================
class Seq2Seq(nn.Module):
    """
    src (B, S) ─ tok_emb + src_pos ─> Encoder × n_layer ─> enc_ln ─> memory (B, S, D)
                                                                        │ cross-attention
    tgt_in (B, T) ─ tok_emb + tgt_pos ─> Decoder × n_layer ─> dec_ln ─> head ─> logits (B, T, V)

    训练（Teacher Forcing）: decoder 的输入是"正确答案右移一位，开头补 <s>"
        tgt     =      7  3  9  1
        tgt_in  = <s>  7  3  9          ← 在位置 t 用 "<s> + 正确答案的前 t 个" 去预测 tgt[t]
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, D)   # 输入、输出共用一张字符表
        self.src_pos = nn.Embedding(cfg.src_len, D)      # 输入的位置编码
        self.tgt_pos = nn.Embedding(cfg.tgt_len, D)      # 输出的位置编码（和输入的分开学）
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layer)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layer)])
        self.enc_ln = nn.LayerNorm(D)
        self.dec_ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.vocab_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.apply(self._init_weights)  # 已写好

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, src):
        """src (B, S) -> memory (B, S, D)"""
        S = src.size(1)
        x = self.tok_emb(src) + self.src_pos(torch.arange(S, device=src.device))   # (B, S, D)
        x = self.drop(x)
        for layer in self.encoder:
            x = layer(x)
        return self.enc_ln(x)   # memory: 给 decoder 的每一层做 cross-attention 用

    def decode(self, tgt_in, memory):
        """tgt_in (B, T), memory (B, S, D) -> logits (B, T, V)，T <= tgt_len"""
        T = tgt_in.size(1)
        y = self.tok_emb(tgt_in) + self.tgt_pos(torch.arange(T, device=tgt_in.device))   # (B, T, D)
        y = self.drop(y)
        for layer in self.decoder:
            y = layer(y, memory, causal=True)   # 每一层都去看同一个 memory
        return self.head(self.dec_ln(y))        # (B, T, V)

    def forward(self, src, tgt):
        """
        训练用（Teacher Forcing）。返回 logits (B, T, V), loss 标量
        """
        B = tgt.size(0)
        # 1. Teacher Forcing: 正确答案右移一位，开头补 <s>
        #      tgt    =      7  3  9  1
        #      tgt_in = <s>  7  3  9
        bos = torch.full((B, 1), BOS, dtype=torch.long, device=tgt.device)
        tgt_in = torch.cat([bos, tgt[:, :-1]], dim=1)

        # 2. 编码一次，解码一次 —— 所有位置的预测并行算出
        logits = self.decode(tgt_in, self.encode(src))

        # 3. 交叉熵: 拍平成 (B*T, V) 和 (B*T,)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, src):
        """
        推理用（自回归 + 贪心）：没有正确答案了，只能把自己上一步的预测当作下一步的输入。
        返回 (B, tgt_len)，不含开头的 <s>
        """
        memory = self.encode(src)   # encoder 只算一次
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)   # (B, 1) 全是 <s>
        for _ in range(self.cfg.tgt_len):
            logits = self.decode(ys, memory)                        # (B, 当前长度, V)
            next_id = logits[:, -1].argmax(-1, keepdim=True)        # 只看最后一位，取概率最大的
            ys = torch.cat([ys, next_id], dim=1)                    # 自己的预测成为下一步的输入
        return ys[:, 1:]   # 去掉开头的 <s>


# =============================================================================
# Step 5: ParallelSeq2Seq（ACT / DETR 风格的并行解码）
# =============================================================================
class ParallelSeq2Seq(Seq2Seq):
    """
    Decoder 的输入不再是"上一步的答案"，而是 tgt_len 个**可学习的查询向量**（query_embed），
    并且 **不用因果掩码**。所以训练和推理都一次就输出全部结果，不需要循环。

        ACT 里: query_embed = nn.Embedding(num_queries=100, hidden_dim)，一次输出 100 步动作

    Encoder 部分直接继承 Seq2Seq 的 encode，不用改。
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__(cfg)
        # tgt_len 个可学习的查询向量，第 i 个负责输出第 i 位（ACT 里是 100 个，负责 100 步动作）
        self.query_embed = nn.Embedding(cfg.tgt_len, cfg.d_model)
        nn.init.normal_(self.query_embed.weight, std=0.02)

    def decode_parallel(self, memory):
        """memory (B, S, D) -> logits (B, tgt_len, V)"""
        B = memory.size(0)
        # 查询向量和输入无关，对每个样本都一样；expand 只是"假装"复制 B 份，不占额外内存
        y = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)   # (B, tgt_len, D)
        for layer in self.decoder:
            y = layer(y, memory, causal=False)   # 不用因果掩码: 所有查询互相可见
        return self.head(self.dec_ln(y))         # (B, tgt_len, V)

    def forward(self, src, tgt):
        # tgt 只用来算 loss，不再作为 decoder 的输入 —— 训练和推理的计算过程完全一样
        logits = self.decode_parallel(self.encode(src))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, src):
        # 一次前向，所有位置同时取 argmax，不需要循环
        return self.decode_parallel(self.encode(src)).argmax(-1)
