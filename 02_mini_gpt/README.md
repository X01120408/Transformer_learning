# 项目 2：字符级 mini-GPT

> 目标：在项目 1 的 `MultiHeadAttention` 基础上，搭出**完整的 Transformer**，
> 真正训练一个能"写莎士比亚"的语言模型。
>
> 预计用时：1~2 天

```
02_mini_gpt/
├── README.md          ← 本讲义（先读这个）
├── model.py           ← 你要填写的代码（所有 TODO）
├── test_model.py      ← 自动测试，不用改
├── prepare_data.py    ← 下载训练数据（已写好）
├── train.py           ← 训练脚本（已写好，建议通读）
└── sample.py          ← 用训练好的模型生成文本（已写好）
```

**开始前：** `conda activate transformer`，然后 `cd 02_mini_gpt`

**强烈推荐配合观看：** Andrej Karpathy《Let's build GPT: from scratch, in code, spelled out》
（YouTube / B 站搬运，约 2 小时）。本项目的结构和他的视频基本一致。

---

## 第一部分：理解 GPT（先读，约 40 分钟）

### 1.1 GPT 只做一件事：预测下一个字符

给定前面的字符，输出"下一个字符是什么"的概率分布。生成文本时，就是不停地"预测 → 抽一个 → 拼到末尾 → 再预测"。

训练数据怎么来？**把同一段文本错开一位**，就得到输入 x 和目标 y：

```
文本:  F  i  r  s  t     C  i  t
x:     F  i  r  s  t     C  i
y:     i  r  s  t     C  i  t
```

一段长 T 的文本一次能提供 **T 个训练样本**：位置 0 用 "F" 预测 "i"，位置 1 用 "Fi" 预测 "r"……
这就是为什么需要项目 1 的**因果掩码**：位置 1 在预测 "r" 时，绝不能看到位置 2 上的 "r"。

### 1.2 把字符变成数字（Tokenization）

本项目用最简单的**字符级**：文本里出现过的每个字符分配一个编号。

```python
chars = sorted(set(text))                  # 65 个字符: '\n', ' ', '!', ..., 'a', ..., 'z'
encode("hi") -> [46, 47]
```

GPT-4 这类大模型用的是 BPE 等"子词"切分，词表有几万到几十万。原理一样，只是一个 token 代表的不再是单个字符。

### 1.3 把数字变成向量（Embedding）

`nn.Embedding(vocab_size, d_model)` 本质上是一张 **vocab_size × d_model 的可学习表格**，编号 i 就取第 i 行。
训练时，意思相近的字符的向量会逐渐靠近。

### 1.4 位置编码：Attention 不知道顺序

回忆项目 1 的思考题 3：**把输入打乱顺序，self-attention 的输出只是跟着打乱，内容不变**。
也就是说，对 attention 来说 "dog bites man" 和 "man bites dog" 没有区别。

解决办法：给每个**位置**也查一张表，把位置向量**加**到字符向量上：

```
x = tok_emb(idx) + pos_emb([0, 1, 2, ..., T-1])
    (B, T, D)        (T, D)  ← 自动广播到每个样本
```

这样同一个字符出现在不同位置时，向量就不一样了。

| 位置编码方式 | 谁在用 |
|---|---|
| 可学习的位置表（本项目） | GPT-2、ViT |
| 固定的正弦 / 余弦函数 | 原始 Transformer、**ACT 的图像特征**（2D 正弦编码） |
| RoPE（旋转位置编码） | LLaMA、Qwen 等现代大模型 |

> ACT 里还有一个 `query_embed`：100 个**可学习的动作查询**向量，它在 decoder 中起的作用
> 和这里的位置编码很像——让第 1 步动作和第 100 步动作能被区分开。

### 1.5 Transformer Block：Attention + FeedForward + 残差 + LayerNorm

```
x ──┬──> LayerNorm ──> 因果 Self-Attention ──(+)──┬──> LayerNorm ──> FeedForward ──(+)──> 输出
    └─────────────── 残差 ─────────────────────┘  └──────────── 残差 ─────────────┘
```

四个部件各自的作用：

| 部件 | 作用 | 一句话 |
|---|---|---|
| **Self-Attention** | 位置之间交换信息 | "开会"：每个字符去收集其他字符的信息 |
| **FeedForward** | 每个位置单独加工信息 | "回工位"：把收集到的信息消化一下 |
| **残差连接** `x + f(x)` | 让梯度能直接流回前面的层 | 深层网络能训练的关键；子层只需学"修改量" |
| **LayerNorm** | 把每个向量归一化（均值 0、方差 1，再缩放） | 稳定数值，训练更稳 |

FeedForward 是 `D → 4D → GELU → D` 的两层 MLP。每个 Block 里**约 2/3 的参数**都在这里。

**Pre-LN 和 Post-LN**：LayerNorm 放在子层之前（本项目，GPT-2 起的主流）还是残差相加之后（原始 Transformer）。
Pre-LN 更容易训练，不太需要调学习率预热。
> **ACT 用的是 Post-LN**：源码里 `normalize_before=False`，激活函数是 ReLU，FFN 维度是 3200。

### 1.6 输出层和损失函数

最后一层 LayerNorm 之后，用一个 `Linear(D → vocab_size)` 给每个字符打分（logits），再用**交叉熵**计算损失。

**一个非常有用的检查：** 刚初始化的模型等于瞎猜，每个字符的概率约为 1/65，
所以初始 loss 应该约等于 **ln(65) ≈ 4.17**。如果你看到的初始 loss 是 10 或 50，说明初始化或代码有问题。

### 1.7 生成：温度和 top-k

```python
logits = logits[:, -1, :] / temperature   # 只用最后一个位置的预测
probs  = softmax(logits)
next   = torch.multinomial(probs, 1)      # 按概率抽一个（不是取最大值）
```

- **temperature**：除以一个数再 softmax。小于 1 时分布变尖锐，结果更保守；大于 1 时分布变平，结果更随机
- **top-k**：只在概率最高的 k 个字符里抽，避免抽到很离谱的字符
- 为什么不直接取最大值（贪心）？因为贪心很容易陷入重复循环，比如 "the the the the ..."

---

## 第二部分：动手实现

每完成一步就跑一次对应的测试，**全绿再往下走**。测试全部在 CPU 上运行，几秒就完成。

| 步骤 | 内容 | 测试命令 | 代码量 |
|---|---|---|---|
| Step 1 | `FeedForward` | `python test_model.py 1` | 2 行 |
| Step 2 | `Block` | `python test_model.py 2` | 约 10 行 |
| Step 3 | `GPT.__init__` 和 `forward` | `python test_model.py 3` | 约 15 行 |
| Step 4 | `GPT.generate` | `python test_model.py 4` | 约 10 行 |

`model.py` 的每个 TODO 都写了详细的步骤提示。测试里有几项特别值得注意：

- **因果性测试**：修改第 t 个位置之后的输入，检查前 t 个位置的输出是否不变。忘了传 mask 就会在这里失败
- **初始 loss 测试**：检查 1.6 节说的 ln(vocab_size)
- **单 batch 过拟合测试**：在同一个小 batch 上训练 150 步，loss 必须接近 0。模型连背下一个 batch 都做不到的话，一定有 bug

---

## 第三部分：训练

```bash
python test_model.py              # 确认 17/17 全部通过
python prepare_data.py            # 下载数据（约 1MB）
python train.py --overfit         # 先做单 batch 过拟合检查（养成习惯！）
python train.py                   # 正式训练
python sample.py --prompt "ROMEO:"
```

训练时每 250 步会打印一次 loss 和一段生成的文本。**仔细看生成的文本是怎么变化的**：
从一堆乱码，到出现空格和换行，到出现像样的单词，再到出现"人名 + 冒号 + 台词"的剧本格式。

参考结果（RTX 3060，默认配置 3.2M 参数，3000 步）：

| | 数值 |
|---|---|
| 训练用时 | 约 2 分钟 |
| 初始 loss | ≈ 4.2（理论值 ln 65 = 4.17） |
| 最终 val loss | ≈ 1.47（train loss ≈ 1.11） |

训练结束后会生成 `loss_curve.png`，打开看看 train 和 val 两条曲线。

---

## 第四部分：动手实验（理解更深的关键）

每个实验跑一次 `python train.py --max_iters 1500 ...`（1~2 分钟），记录 val loss，对比默认配置。

| # | 实验 | 命令 | 想一想 |
|---|---|---|---|
| 1 | 温度 | `python sample.py --temperature 0.3`，再试 `1.5` | 生成结果有什么区别？ |
| 2 | 去掉位置编码 | 在 `forward` 里临时删掉 `+ self.pos_emb(pos)` | loss 变差多少？生成的文本有什么问题？ |
| 3 | 上下文长度 | `--block_size 16` 对比 `--block_size 256` | 能看得更远有什么好处？ |
| 4 | 模型大小 | `--n_layer 1` 对比 `--n_layer 6 --d_model 384 --n_head 6` | 大模型的 train 和 val loss 差距是否变大？（过拟合） |
| 5 | 中文数据 | `python prepare_data.py --from 你的文本.txt` | 词表变大后，初始 loss 是多少？ |

---

## 常见坑

1. **`self.attn(...)` 返回两个值**：项目 1 的 `MultiHeadAttention` 返回 `(out, weights)`，要写 `a, _ = self.attn(...)`
2. **mask 的 device**：数据在 GPU 上而 mask 在 CPU 上会报错，所以要传 `device=x.device`
3. **pos 的 device**：`torch.arange(T)` 默认在 CPU，同样要加 `device=idx.device`
4. **cross_entropy 的形状**：它要求 `(N, C)` 和 `(N,)`，要先把 `(B, T, V)` 拍平成 `(B*T, V)`
5. **generate 里忘了裁剪上下文**：生成超过 block_size 个字符后，`pos_emb` 会越界报错
6. **generate 用了全部位置的 logits**：只有最后一个位置的输出才是"下一个字符"的预测
7. **评估时忘了 `model.eval()`**：dropout 没关掉，loss 会偏高（train.py 里已经处理了）

## 思考题

1. 默认配置下，模型的参数大约有多少在 FeedForward 里？多少在 Attention 里？（可以用测试里的参数量公式算）
2. 训练时一个 batch 一次就能算出所有位置的 loss，那**生成**时为什么只能一个字符一个字符地来？
3. 生成第 1000 个字符时，前面 999 个字符的 K 和 V 每次都被重新计算了一遍。有什么办法避免重复计算？（提示：搜索 KV Cache）
4. 实验 4 中，如果大模型的 train loss 远低于 val loss，说明了什么？可以怎么缓解？
5. **联系 ACT**：GPT 是一个字符一个字符地生成，ACT 却能一次输出 100 步动作。两者在结构上的哪些差别导致了这一点？
   （提示：因果掩码、decoder 的输入是什么、输出头预测的是什么、loss 用的是什么）
