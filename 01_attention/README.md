# 项目 1：手写 Attention

> 目标：不调库，自己写出 Transformer 最核心的模块 —— Attention 和 Multi-Head Attention，
> 并且做到和 PyTorch 官方实现**输出逐位一致**。
>
> 预计用时：半天到 1 天

```
01_attention/
├── README.md            ← 本讲义（先读这个）
├── attention.py         ← 你要填写的代码（所有 TODO）
├── test_attention.py    ← 自动测试，不用改
└── visualize.py         ← 全部通过后运行，画注意力热力图
```

**开始前：** `conda activate transformer`，然后 `cd 01_attention`

---

## 第一部分：理解 Attention（先读，约 30 分钟）

### 1.1 一句话直觉：一种"模糊查字典"

普通的字典（Python dict）：拿一个 **query** 去和所有 **key** 精确比对，命中哪个就返回哪个 **value**。

Attention 是"模糊版"：
1. 拿 query 和**每一个** key 算相似度（打分）
2. 把分数变成概率（softmax），相似度越高权重越大
3. 返回所有 value 的**加权平均**

```
query ──比较──> key_1  key_2  key_3  key_4
相似度:          2.1    0.3    5.0    -1.0
softmax权重:     0.05   0.01   0.94   0.00     ← 和为 1
输出 = 0.05*value_1 + 0.01*value_2 + 0.94*value_3 + 0.00*value_4
```

在句子里，每个词都会同时扮演 query、key、value 三种角色：
"每个词去看看句子里其他所有词，按相关程度把它们的信息吸收进来"。

> **联系 ACT**：ACT 的 decoder 有 100 个"动作查询"（query），每个查询去关注
> 图像特征 + 关节状态（key/value），从中提取"这一步动作需要的信息"。

### 1.2 公式

$$
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**一定要盯着形状看**（这是学 Transformer 最重要的习惯）：

| 量 | 形状 | 含义 |
|---|---|---|
| Q | (L, d_k) | L 个 query，每个是 d_k 维向量 |
| K | (S, d_k) | S 个 key（维度必须和 Q 一样，才能做点积） |
| V | (S, d_v) | S 个 value，和 key 一一对应 |
| Q Kᵀ | (L, S) | 第 i 行第 j 列 = 第 i 个 query 和第 j 个 key 的点积相似度 |
| softmax(...) | (L, S) | 对**每一行**做 softmax，每行和为 1 |
| 输出 | (L, d_v) | 每个 query 得到一个加权平均后的 value |

实际代码中前面还会多出 batch 维、head 维，比如 (B, H, L, d_k)。
但矩阵乘法 `@` 只作用于**最后两维**，所以公式写法完全不变。

### 1.3 为什么要除以 √d_k？

点积 q·k 是 d_k 个乘积之和。如果 q、k 的每个分量都是均值 0、方差 1，
那么点积的方差是 **d_k**。d_k=64 时，分数的标准差约为 8。

分数太大会怎样？softmax 会变得极其"尖锐"，几乎变成 one-hot：
- 输出只取一个 value，失去"加权平均"的意义
- 梯度几乎为 0，模型很难训练

除以 √d_k 后，分数的方差回到 1。**做完 Step 1 后自己验证一下**（见思考题 2）。

### 1.4 Mask（掩码）：不让模型看某些位置

做法：在 softmax **之前**，把不允许看的位置的分数设成 **-∞**。
因为 e^(-∞) = 0，softmax 之后这些位置的权重就严格为 0，其余位置的权重仍然和为 1。

两种常见用途：
- **Causal mask（因果掩码）**：生成文本时，第 i 个词只能看第 0..i 个词，不能"偷看未来"。
  这是一个下三角矩阵（Step 2）。GPT 用的就是它。
- **Padding mask**：一个 batch 里句子长短不一，短句会补零（padding），不能让模型关注这些补出来的位置。

本项目约定：**mask 中 True = 允许看，False = 屏蔽**（与 `F.scaled_dot_product_attention` 一致）。

### 1.5 Multi-Head Attention：多个头，各看各的

只有一组 Q/K/V 时，每个词只能"按一种标准"去关注别人。
多头的想法是：把 d_model 维切成 H 份，每份（d_head = d_model / H 维）独立做 attention。
这样不同的头可以学会关注不同的关系，比如一个头关注语法，另一个头关注指代。

```
x (B, T, D)
  │  W_q / W_k / W_v   三个线性层（可学习参数就在这里）
  ▼
(B, T, D)
  │  split_heads: view 成 (B, T, H, Dh)，再 transpose 成 (B, H, T, Dh)
  ▼
(B, H, T, Dh)    ← H 个头像 batch 一样并行计算，不用写 for 循环
  │  scaled_dot_product_attention（Step 1 写的函数）
  ▼
(B, H, T, Dh)
  │  merge_heads: transpose 回 (B, T, H, Dh)，再 reshape 成 (B, T, D)
  ▼
(B, T, D)
  │  W_o   输出线性层，把各个头的信息混合起来
  ▼
out (B, T, D)
```

注意：多头**不增加计算量**。H 个 Dh 维的头，总计算量和 1 个 D 维的头差不多。

### 1.6 Self-Attention 和 Cross-Attention

同一个 `MultiHeadAttention` 模块，只是输入不同：

| | query 来自 | key / value 来自 | 例子 |
|---|---|---|---|
| Self-Attention | x | x（自己） | GPT、ACT 的 encoder |
| Cross-Attention | 序列 A | 序列 B | ACT 的 decoder：动作查询去关注图像特征 |

Cross-attention 时 L（query 长度）和 S（key 长度）可以不同，测试里也专门测了这种情况。

---

## 第二部分：动手实现

每完成一步就跑一次对应的测试。**测试全绿再往下走。**

### Step 1：`scaled_dot_product_attention`

```bash
python test_attention.py 1
```

照着 `attention.py` 里的 TODO(1a)~(1d) 写，4 到 5 行代码。

### Step 2：`make_causal_mask`

```bash
python test_attention.py 2
```

1 行代码。写完后在 Python 里打印 `make_causal_mask(5)`，亲眼看看它长什么样。

### Step 3：`MultiHeadAttention`

```bash
python test_attention.py 3
```

按 3a → 3b → 3c → 3d 的顺序写。split_heads 和 merge_heads 是最容易出错的地方，
建议先在 Python 交互环境里拿一个小张量试一试：

```python
import torch
x = torch.arange(2*3*8).float().view(2, 3, 8)   # B=2, T=3, D=8，准备切成 H=2 个头
print(x[0])
print(x.view(2, 3, 2, 4).transpose(1, 2)[0])    # 看看第 0 个样本被切成了什么样
```

### 全部通过后

```bash
python test_attention.py     # 应显示 通过 12/12
python visualize.py          # 生成 attention_vis.png，打开看看，回答它打印的 3 个问题
```

---

## 测试结果怎么看

| 标记 | 含义 |
|---|---|
| ✅ | 通过 |
| ⏸ | 还没实现（还有 `raise NotImplementedError` 没删） |
| ❌ | 结果不对，下面会提示哪里错了（形状？数值？） |
| 💥 | 代码报错，会显示出错的是 attention.py 的哪一行 |

## 常见坑

1. **softmax 的维度**：应该在最后一维（对每个 query，在所有 key 上归一化），即 `dim=-1`
2. **mask 填 0 而不是 -inf**：分数为 0 经过 softmax 后权重并不是 0（e⁰ = 1）
3. **mask 的方向**：`masked_fill` 填的是条件为 True 的位置，而我们要屏蔽的是 mask 为 False 的位置
4. **merge_heads 时忘了 transpose**：直接 `view(B, T, D)` 形状对了，但数据是乱的
5. **transpose 之后用 view 报错**：`view size is not compatible...`，改用 `reshape`
6. **写死了维度**：比如用 `k.transpose(2, 3)`，遇到 3 维输入就错了。要用负数索引 `transpose(-2, -1)`

## 思考题（做完后回答，可以来问我）

1. Attention 本身（Step 1，不算 W_q 等线性层）有可学习参数吗？那模型"学"的到底是什么？
2. **动手实验**：令 `q = torch.randn(1, 10, 512)`、`k = torch.randn(1, 10, 512)`，
   分别计算除以 √512 和不除的 softmax 权重，打印出来对比。你看到了什么？
3. 如果把输入序列的词打乱顺序，self-attention 的输出会怎样变化？
   这说明 Transformer 还缺少什么？（提示：项目 2 会解决这个问题）
4. 序列长度 L 翻倍时，attention 的计算量和显存会变成原来的几倍？为什么？
5. 在 ACT 中，decoder 的 query 是什么？key 和 value 又是什么？
   （可以打开你之前跑的 ACT 代码里的 `detr/models/transformer.py` 找答案）

## 推荐阅读（配合本项目）

- Jay Alammar《The Illustrated Transformer》—— 图解，有中文翻译
- 3Blue1Brown《Attention in transformers, visually explained》—— B 站有中文字幕
- 原论文《Attention Is All You Need》第 3.2 节 —— 只有一页，做完本项目后再看会很容易
