# 项目 3：Encoder-Decoder Transformer（Seq2Seq）

> 目标：搭出原始论文《Attention Is All You Need》里的**完整 Transformer**（Encoder + Decoder），
> 理解 **cross-attention** 怎样把两个序列连起来；
> 再实现一个 **ACT 风格的并行解码器**，对比两种解码方式。
>
> 预计用时：1~2 天

```
03_seq2seq/
├── README.md          ← 本讲义（先读这个）
├── seq2seq.py         ← 你要填写的代码（所有 TODO）
├── test_seq2seq.py    ← 自动测试，不用改
├── data.py            ← 随机生成训练数据（已写好，不需要下载）
└── train.py           ← 训练 + 画 cross-attention 热力图（已写好）
```

**开始前：** `conda activate transformer`，然后 `cd 03_seq2seq`，先运行 `python data.py` 看看三个任务长什么样。

---

## 第一部分：理解 Encoder-Decoder（先读，约 40 分钟）

### 1.1 为什么需要两部分

项目 2 的 GPT 只有一个序列：用前文预测后文。但很多任务是**把一个序列变成另一个序列**：

| 任务 | 输入序列 | 输出序列 |
|---|---|---|
| 翻译 | 英文句子 | 中文句子 |
| 本项目 | `3 1 4 1 5` | `5 1 4 1 3`（倒序）|
| **ACT** | 图像特征 + 关节状态 | 未来 100 步的关节动作 |

Encoder-Decoder 的分工：
- **Encoder**："读懂"输入，把它变成一组向量 memory
- **Decoder**：一边生成输出，一边通过 **cross-attention** 去 memory 里"查资料"

```
            src: 3 1 4 1 5
                   │
          ┌────────▼────────┐
          │  Encoder × N    │  双向 self-attention：每个位置看所有位置
          └────────┬────────┘
                   │ memory (B, S, D)
                   │
          ┌────────▼────────┐
tgt_in ──►│  Decoder × N    │  ① self-attention（输出序列内部交流）
          │                 │  ② cross-attention（q 来自 decoder，k/v 来自 memory）
          │                 │  ③ FeedForward
          └────────┬────────┘
                   ▼
             logits → 5 1 4 1 3
```

### 1.2 Encoder：双向的

Encoder 层和项目 2 的 Block **几乎一模一样，只是不传因果掩码**。
因为输入是一次性全部给出的，不存在"偷看未来"的问题；让每个位置看到整个输入，理解得更充分。
（BERT 就是只有 Encoder 的模型。）

### 1.3 Decoder：多了 Cross-Attention

Decoder 层有 **3 个子层**，每个都带残差：

```python
y = y + SelfAttention(LN1(y), mask)            # ① 输出序列内部交流
y = y + CrossAttention(q=LN2(y), k=v=memory)    # ② 去 encoder 的输出里查信息  ← 新东西
y = y + FeedForward(LN3(y))                     # ③ 加工
```

Cross-attention 用的就是项目 1 你写的 `MultiHeadAttention`，只是传参不同：

```python
self.cross_attn(h, memory, memory)   # query = decoder 的 h (B,T,D)，key = value = memory (B,S,D)
```

注意力权重的形状是 **(B, H, T, S)**：输出的第 t 位，对输入每一位的关注程度。
训练完画出来，就能看到"输出的每一位在看输入的哪里"。倒序任务的热力图是一条清楚的**反对角线**。

### 1.4 训练和推理不一样：Teacher Forcing

**训练时**，我们知道正确答案，把"正确答案右移一位、开头补 `<s>`"喂给 decoder，配合因果掩码，**一次前向就能并行算出所有位置的 loss**：

```
tgt     =       7   3   9   1
tgt_in  = <s>   7   3   9            ← decoder 的输入
位置 0: 看到 <s>          → 预测 7
位置 1: 看到 <s> 7        → 预测 3
位置 2: 看到 <s> 7 3      → 预测 9
```

这叫 **Teacher Forcing**（老师手把手喂正确答案）。

**推理时**，没有正确答案了，只能**把自己上一步的预测当作下一步的输入**，一位一位地生成（自回归），和项目 2 的 generate 一样：

```
<s>          → 预测 7
<s> 7        → 预测 3
<s> 7 3      → 预测 9 ...
```

> 一个隐患：训练时 decoder 看到的永远是正确答案，推理时却要看自己的预测。
> 一旦某一步预测错了，后面的输入就是模型训练时没见过的情况，错误可能越滚越大（exposure bias）。

### 1.5 另一种解码方式：ACT / DETR 的并行解码

ACT 不是一位一位地生成动作的。它的 decoder：

1. 输入不是"上一步的答案"，而是 **100 个可学习的查询向量**（`query_embed`），每个查询负责输出一个时间步的动作
2. **不用因果掩码**，100 个查询之间可以互相看
3. **训练和推理都一次前向就输出全部结果**，不需要循环

| | 自回归解码（标准 Transformer、GPT） | 并行解码（ACT、DETR） |
|---|---|---|
| decoder 的输入 | `<s>` + 之前的输出 | 可学习的查询向量 `query_embed` |
| 因果掩码 | 有 | **没有** |
| 推理时前向次数 | 输出多长就算多少次 | **1 次** |
| 训练和推理是否一致 | 不一致（teacher forcing） | 一致 |
| 输出之间的依赖 | 后面的输出能直接看到前面输出的**值** | 只能通过 self-attention 在**向量层面**协调 |

本项目的 Step 5 会实现并行解码，你可以在同样的任务上直接对比两者。

### 1.6 对照 ACT

| 本项目 | ACT（你复现的代码 `detr/models/transformer.py`） |
|---|---|
| src = 数字 token + 位置编码 | src = [隐变量 z, 关节状态, ResNet 图像特征] + 正弦位置编码 |
| Encoder 2 层 | Encoder 4 层 |
| `ParallelSeq2Seq.query_embed`（tgt_len 个） | `query_embed`（chunk_size = 100 个） |
| Decoder 2 层，`causal=False` | Decoder 7 层，调用时不传 `tgt_mask` |
| head：Linear → 12 个字符的打分，交叉熵 | head：Linear → 14 维关节角，**L1 回归损失** |
| Pre-LN | Post-LN（`normalize_before=False`） |

一个小区别：ACT 的 decoder 输入是全 0 的 `tgt`，`query_embed` 作为位置编码在**每一层**加到 q 和 k 上；
本项目简化为直接把 `query_embed` 当作 decoder 的输入。思路是一样的。

---

## 第二部分：动手实现

| 步骤 | 内容 | 测试命令 | 代码量 |
|---|---|---|---|
| Step 1 | `EncoderLayer` | `python test_seq2seq.py 1` | 约 8 行（照搬项目 2 的 Block，去掉 mask） |
| Step 2 | `DecoderLayer` | `python test_seq2seq.py 2` | 约 15 行 |
| Step 3 | `Seq2Seq` 的 `__init__`、`encode`、`decode`、`forward` | `python test_seq2seq.py 3` | 约 25 行 |
| Step 4 | `Seq2Seq.generate`（自回归） | `python test_seq2seq.py 4` | 约 6 行 |
| Step 5 | `ParallelSeq2Seq`（ACT 风格） | `python test_seq2seq.py 5` | 约 10 行 |

几项值得留意的测试：
- **双向测试**（Step 1）：修改最后一个位置，第一个位置的输出必须变。和项目 2 的因果性测试正好相反
- **Teacher Forcing 测试**（Step 3）：修改 `tgt[t:]`，位置 0~t 的 logits 不能变。忘了"右移一位"就会在这里失败——模型在预测 `tgt[t]` 时看到了 `tgt[t]` 本身，等于抄答案
- **贪心自洽测试**（Step 4）：把 generate 的输出当作答案喂回 forward，每个位置的 argmax 应该正好是 generate 选的字符
- **不使用 tgt 测试**（Step 5）：并行解码时 tgt 只用来算 loss，改变 tgt 不应影响 logits

---

## 第三部分：训练

```bash
python test_seq2seq.py                              # 确认 25/25 全部通过
python train.py --task reverse                      # 倒序
python train.py --task sort                         # 排序
python train.py --task add                          # 加法
python train.py --task add --decoder parallel       # 换成 ACT 风格的并行解码
```

每次训练约 **25 秒**（RTX 3060，3000 步，0.8M 参数）。每一步都用全新随机生成的数据，所以不存在"背答案"，测试准确率就是真实水平。

参考结果（3000 步，"整体准确率"= 整串全对的比例）：

| 任务 | 自回归 | 并行（ACT 风格） | 难点 |
|---|---|---|---|
| reverse 倒序 | 100% | 100% | 只需要按位置对应地"抄" |
| sort 排序 | 100% | 100% | 要统计每个数字出现几次 |
| add 加法 | 约 97% | 100% | 进位：一位的结果取决于低位 |

加法是三个任务里最难的：500 步时整体准确率还不到 1%，3000 步才接近全对。
这是我跑的一次结果，换个随机种子可能略有不同。

训练结束会保存 `cross_attn_<任务>_<解码方式>.png`，画的是 decoder 每一层的 cross-attention（对所有头取平均）。**一定要打开看看**：
- **reverse**：第 1 层是一条清楚的**反对角线**，输出第 1 位看输入最后一位
- **sort**：输出某个数字时，在看输入里哪些位置？
- **add**：注意力会集中到少数几个位置上，比如 `+` 号。想想为什么（思考题 4）

---

## 第四部分：动手实验

| # | 实验 | 做法 | 想一想 |
|---|---|---|---|
| 1 | 两种解码方式对比 | 3 个任务 × 2 种解码方式都跑一遍 | 哪种收敛更快？准确率差多少？ |
| 2 | 去掉输入的位置编码 | 在 `encode` 里临时删掉 `+ self.src_pos(...)` | reverse 还能学会吗？sort 呢？为什么差别这么大？ |
| 3 | 模型更小 | `--n_layer 1`，再试 `--d_model 32` | 哪个任务最先学不会？ |
| 4 | 训练更短 | `--steps 500` | 加法的字符准确率和整体准确率差多少？为什么整体准确率低这么多？ |
| 5 | 比较热力图 | 对比同一任务两种解码方式的热力图 | 并行解码的注意力模式有什么不同？ |

---

## 常见坑

1. **Encoder 误加了因果掩码**：Encoder 应该是双向的，`self.attn(h, h, h)` 不传 mask
2. **cross-attention 传错参数**：应该是 `self.cross_attn(h, memory, memory)`，不是 `(h, h, h)`，也不是 `(memory, h, h)`。query 决定输出有几个位置，所以 query 必须来自 decoder
3. **忘了右移一位**：直接把 `tgt` 喂给 decoder，训练 loss 会降到几乎为 0，但 generate 的结果一塌糊涂
4. **generate 返回时没去掉开头的 `<s>`**：输出长度会多 1
5. **generate 里每一步都重新 encode**：结果没错，但很浪费。encoder 只需要算一次
6. **并行解码忘了 `causal=False`**：测试会抓到。虽然也能训练，但前面的查询看不到后面的查询，协调能力变弱
7. **`expand` 和 `repeat`**：`expand` 不复制内存，只是"假装"有 B 份，比 `repeat` 省内存

## 思考题

1. Encoder 为什么不需要因果掩码，Decoder 的 self-attention 为什么需要？并行解码时又为什么不需要了？
2. 训练时用 teacher forcing，推理时却要看自己的预测。这种不一致会带来什么问题？并行解码为什么没有这个问题？
3. 并行解码时，如果把 decoder 的 self-attention 去掉，只保留 cross-attention，排序任务还能做好吗？（提示：输入里有重复的数字时，输出的相邻几位怎么"商量"谁输出哪个？）
4. 加法任务的 cross-attention 为什么会集中在 `+` 号上，而不是两个加数对应的位？（提示：encoder 有 2 层 self-attention，到 memory 的时候，`+` 号这个位置的向量里已经装了什么信息？）
5. **联系 ACT**：机器人控制为什么更适合用并行解码，一次输出整段动作？从推理速度、动作的平滑性、误差累积三个角度想一想。
