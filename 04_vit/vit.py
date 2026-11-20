"""
项目 4：Vision Transformer（ViT）
================================

你的任务：补完所有 TODO，并删掉对应的 `raise NotImplementedError`。

建议顺序：
    Step 1  patchify（把图片切成小块）     -> python test_vit.py 1
    Step 2  PatchEmbedding（小块 -> 向量） -> python test_vit.py 2
    Step 3  ViT（CLS token + 位置编码 + Encoder + 分类头） -> python test_vit.py 3
    全部通过后                              -> python train.py

复用的模块:
    项目 3: EncoderLayer（双向 self-attention，内部又用到项目 1 的 MultiHeadAttention、项目 2 的 FeedForward）
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_seq2seq"))
from seq2seq import EncoderLayer  # noqa: E402


@dataclass
class ViTConfig:
    img_size: int = 32       # 输入图片边长（CIFAR-10 是 32×32）
    patch_size: int = 4      # 每个小块的边长 -> (32/4)² = 64 个 patch
    in_chans: int = 3        # RGB 三通道
    num_classes: int = 10
    d_model: int = 192
    n_layer: int = 6
    n_head: int = 3
    dropout: float = 0.0

    @property
    def n_patches(self):
        return (self.img_size // self.patch_size) ** 2


# =============================================================================
# Step 1: patchify —— 把图片切成不重叠的小块，每块拉平成一个向量
# =============================================================================
def patchify(img, P):
    """
    img: (B, C, H, W)  ->  (B, N, C*P*P)，其中 N = (H/P) * (W/P)

    例: (B, 3, 32, 32)，P = 4  ->  (B, 64, 48)

    顺序约定（测试会检查）:
        - patch 按行优先排列: 第 0 个是左上角，第 1 个在它右边……第 W/P 个是第二行的第一个
        - 每个 patch 内部按 (C, P, P) 的顺序拉平: 先 R 通道的 16 个像素，再 G，再 B
          （和 Conv2d 的权重形状 (D, C, P, P) 一致，Step 2 会用到这一点）

    思路: 用 reshape 把 H 拆成 (H/P, P)、W 拆成 (W/P, P)，再用 permute 把维度换到想要的顺序。
    """
    B, C, H, W = img.shape
    assert H % P == 0 and W % P == 0, "图片边长必须能被 patch_size 整除"
    h, w = H // P, W // P

    # 1. 把 H 拆成 (h, P)、W 拆成 (w, P)。只改形状，不动数据:
    #    img[b, c, i*P + p, j*P + q]  ==  x[b, c, i, p, j, q]
    x = img.reshape(B, C, h, P, w, P)
    # 2. 调整维度顺序: 把"第几个 patch"(h, w) 放前面，"patch 内部"(C, P, P) 放后面
    #    (B, C, h, P, w, P) -> (B, h, w, C, P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    # 3. 合并: (h, w) -> N 个 patch（行优先），(C, P, P) -> 每个 patch 拉平成 C*P*P 维
    return x.reshape(B, h * w, C * P * P)


# =============================================================================
# Step 2: Patch Embedding —— 每个小块线性映射成一个 D 维向量
# =============================================================================
class PatchEmbedding(nn.Module):
    """
    概念上: patchify 之后接一个 nn.Linear(C*P*P, D)。
    实际中: 用一个 kernel_size = stride = P 的 Conv2d 一步完成 —— 两者在数学上完全等价
           （卷积核每次正好盖住一个 patch，且不重叠），但 Conv2d 更快、写法更简洁。

        (B, C, H, W) --Conv2d--> (B, D, h, w) --flatten--> (B, D, N) --transpose--> (B, N, D)
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        # kernel = stride = P: 卷积核每次正好盖住一个 patch，且互不重叠
        # 权重形状 (D, C, P, P)，相当于 Linear(C*P*P, D) 的权重
        self.proj = nn.Conv2d(cfg.in_chans, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size)

    def forward(self, img):
        """(B, C, H, W) -> (B, N, D)"""
        x = self.proj(img)        # (B, D, h, w)    每个 patch 变成一个 D 维向量
        x = x.flatten(2)          # (B, D, h*w)     把二维的 patch 网格拉平成一维（行优先）
        return x.transpose(1, 2)  # (B, N, D)       和 NLP 一样: (batch, token 数, 向量维度)


# =============================================================================
# Step 3: ViT
# =============================================================================
class ViT(nn.Module):
    """
    img (B, 3, 32, 32)
      │ PatchEmbedding                                   (B, 64, D)
      │ 在最前面拼上一个可学习的 [CLS] token              (B, 65, D)
      │ + 可学习的位置编码 pos_embed                       (1, 65, D) 广播
      │ EncoderLayer × n_layer（项目 3，双向，无 mask）     (B, 65, D)
      │ LayerNorm
      │ 只取第 0 个位置（CLS）的输出                        (B, D)
      │ head: Linear(D -> num_classes)
      ▼
    logits (B, 10)

    CLS token: 一个不对应任何 patch 的"空白"向量。经过多层 self-attention 后，
    它从所有 patch 收集信息，最后用它的输出来做分类。（来自 BERT 的做法）
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg)
        # nn.Parameter: 直接把一个张量注册为可学习参数（不需要像 Embedding 那样"查表"）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))                   # 1 个汇总用的 token
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.n_patches + 1, cfg.d_model))   # 64 个 patch + 1 个 CLS
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layer)])  # 项目 3: 双向，无 mask
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        self._init_weights()  # 已写好

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img):
        """img (B, C, H, W) -> logits (B, num_classes)"""
        B = img.size(0)
        # 1. 图片 -> patch token
        x = self.patch_embed(img)                      # (B, 64, D)
        # 2. 在最前面拼上 CLS（所有图片共用同一个 CLS 向量，expand 到 batch）
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                 # (B, 65, D)
        # 3. 加位置编码 (1, 65, D)，广播到每张图
        x = self.drop(x + self.pos_embed)
        # 4. 双向 self-attention: 每个 patch 和 CLS 都能看到所有 token
        for block in self.blocks:
            x = block(x)
        # 5-6. 只用 CLS 位置的输出做分类
        x = self.ln(x)
        return self.head(x[:, 0])                      # (B, 10)
