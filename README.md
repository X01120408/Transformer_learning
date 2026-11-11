# Transformer 学习路线

从零开始、以动手为主学 Transformer，最终目标是能**完全看懂并自己重写 ACT**。

## 环境

```bash
conda activate transformer     # Python 3.11, torch 2.15 + CUDA 13.2, RTX 3060
```

## 进度

| # | 项目 | 学到什么 | 状态 |
|---|------|---------|------|
| 1 | [手写 Attention](01_attention/) | QKV、缩放、mask、多头、self / cross attention | 🔨 进行中 |
| 2 | 字符级 mini-GPT | Decoder-only 结构、位置编码、LayerNorm、残差、训练循环 | ⬜ |
| 3 | Seq2Seq 小任务 | 完整的 Encoder-Decoder、cross-attention 的实际作用 | ⬜ |
| 4 | 从零写 ViT（CIFAR-10） | 把图像切成 patch 变成 token | ⬜ |
| 5 | Mini-ACT | CVAE、action chunking、temporal ensembling | ⬜ |

## 学习习惯

1. 每一层都打印 shape
2. 先在一个小 batch 上把 loss 训到接近 0，再正式训练
3. 先用小模型跑通，再加大规模
4. 画 attention 热力图，看模型在关注什么
5. 自己写完后再对照官方实现，看看哪里不一样、为什么
