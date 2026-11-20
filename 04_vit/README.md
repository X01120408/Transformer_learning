# 项目 4：Vision Transformer（ViT）

> 目标：把**图像**变成 token 序列，用 Transformer 做图像分类（CIFAR-10，10 类）。
> 理解 patch embedding、CLS token、二维图像的位置编码，并和 CNN 做对比。
>
> 预计用时：1 天

```
04_vit/
├── README.md          ← 本讲义（先读这个）
├── vit.py             ← 你要填写的代码（所有 TODO）
├── test_vit.py        ← 自动测试，不用改，不需要数据
├── prepare_data.py    ← 下载 CIFAR-10（已写好）
├── train.py           ← 训练（已写好，可以训练 ViT 或对照组 CNN）
└── visualize.py       ← 可视化 patch、位置编码、注意力（已写好）
```

**开始前：** `conda activate transformer`，然后 `cd 04_vit`

---

## 第一部分：理解 ViT（先读，约 30 分钟）

### 1.1 核心想法：一张图 = 一串"词"

Transformer 处理的是 token 序列。文本天然是一串字符，图像怎么办？

ViT（论文《An Image is Worth 16x16 Words》，2020）的做法非常直接：
**把图片切成一个个不重叠的小方块（patch），每个 patch 当作一个 token。**

```
32×32 的图片，patch = 4×4

┌──┬──┬──┬──┬──┬──┬──┬──┐
│ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│      8 × 8 = 64 个 patch
├──┼──┼──┼──┼──┼──┼──┼──┤      每个 patch: 3 通道 × 4 × 4 = 48 个数
│ 8│ 9│10│..│  │  │  │15│      拉平成一个 48 维向量
├──┼──┼──┼──┼──┼──┼──┼──┤
│  ...                  │      → 一张图变成 64 个 token，形状 (64, 48)
└──┴──┴──┴──┴──┴──┴──┴──┘
```

之后的事情和项目 3 的 Encoder 完全一样：线性映射到 D 维 → 加位置编码 → 若干层双向 self-attention。

### 1.2 Patch Embedding：用 Conv2d 实现

每个 patch（48 维）要经过一个 `Linear(48, D)` 变成 D 维向量。

实际代码里通常用一个 **`kernel_size = stride = P` 的 Conv2d** 一步完成：卷积核每次正好盖住一个 patch，而且不重叠，
所以它和"切块 + Linear"在数学上**完全等价**（测试里会验证这一点）。

```
(B, 3, 32, 32) --Conv2d(3, D, kernel=4, stride=4)--> (B, D, 8, 8) --flatten + transpose--> (B, 64, D)
```

> **联系 ACT**：ACT 的图像也是变成了一串 token，只是不用 patch 切块，而是先过 **ResNet18**：
> 480×640 的图片 → ResNet → 512 通道 × 15×20 的特征图 → 1×1 卷积降到 hidden_dim → 拉平成 300 个 token。
> 也就是说，ACT 里每个"图像 token"对应原图上一块 32×32 的区域，只是这块区域的特征是 CNN 提取的。

### 1.3 CLS token：一个专门负责"汇总"的 token

分类只需要一个输出，但 Encoder 输出了 64 个向量，用哪个？

ViT 的做法（来自 BERT）：在序列最前面**额外加一个可学习的向量 [CLS]**，它不对应任何 patch。
经过多层 self-attention，CLS 会从所有 patch 收集信息，最后**只用它的输出做分类**。

```
[CLS] p0 p1 p2 ... p63        ← 65 个 token
  │
  ▼ Encoder × 6
  │
[CLS'] ...                     ← 只取 CLS' → Linear → 10 个类别的打分
```

另一种常见做法是把 64 个输出取平均（global average pooling），效果差不多。

> **联系 ACT**：ACT 的 CVAE 编码器里也有一个 **CLS token**：把 [CLS, 关节状态, 100 步动作] 送进 Transformer，
> 用 CLS 的输出预测隐变量 z 的均值和方差。和这里的用法一模一样。

### 1.4 位置编码：二维的位置

项目 3 刚验证过：没有位置编码，attention 就分不清顺序。对图像来说，就是分不清 patch 在画面的哪里。

ViT 用最简单的方法：**65 个可学习的位置向量**（`nn.Parameter`），直接加到 token 上。
它只是一维的编号 0~64，模型并不知道"第 8 个 patch 在第 0 个的正下方"。

神奇的是，训练后**位置编码自己学会了二维结构**：相邻 patch 的位置向量很相似，同一行、同一列的也比较相似。
`visualize.py` 会把它画出来。

| 位置编码 | 谁在用 |
|---|---|
| 可学习，一维编号 | ViT（本项目）、GPT-2 |
| **固定的 2D 正弦编码**（行、列各占一半维度） | **ACT、DETR** |

### 1.5 ViT 和 CNN：归纳偏置

CNN 天生带着两个关于图像的"先验假设"（**归纳偏置**）：
- **局部性**：卷积核只看一小块邻域，相邻像素关系最紧密
- **平移不变性**：同一个卷积核在整张图上滑动，猫在左上角还是右下角，用的是同一套特征

ViT 几乎没有这些假设：从第一层开始，每个 patch 就能看到所有 patch，相邻关系要靠位置编码**从数据里学**。

结果是：
- **数据少时，CNN 更好**：先验假设帮它少走弯路
- **数据多时（几千万张图以上），ViT 更好**：不被假设束缚，上限更高

CIFAR-10 只有 5 万张图，属于"数据少"。所以本项目里 ViT 打不过参数更少的 CNN，这是正常的，
训练脚本提供了一个 CNN 对照组让你亲眼看到。大量的数据增强能部分弥补 ViT 的劣势。

---

## 第二部分：动手实现

| 步骤 | 内容 | 测试命令 | 代码量 |
|---|---|---|---|
| Step 1 | `patchify`：切块 | `python test_vit.py 1` | 3 行 |
| Step 2 | `PatchEmbedding`：Conv2d | `python test_vit.py 2` | 4 行 |
| Step 3 | `ViT`：CLS + 位置编码 + Encoder + 分类头 | `python test_vit.py 3` | 约 15 行 |

Step 1 是这次最需要动脑的地方：用 `reshape` + `permute` 把图片切块，要想清楚每个维度最后去了哪里。
建议先在 Python 里用一张小图试一试：

```python
import torch
img = torch.arange(16).view(1, 1, 4, 4)       # 1 通道、4×4 的"图片"，像素值就是编号
print(img[0, 0])
x = img.reshape(1, 1, 2, 2, 2, 2)             # P = 2，切成 2×2 个 patch
print(x.permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 4))   # 每一行应该是一个 2×2 的 patch
```

Encoder 层直接导入项目 3 写的 `EncoderLayer`。到这里，你前三个项目写的代码全部用上了。

---

## 第三部分：训练

```bash
python test_vit.py                  # 确认 13/13 全部通过（不需要数据）
python prepare_data.py              # 下载 CIFAR-10（约 144MB，国内镜像）
python train.py                     # 训练 ViT，30 个 epoch
python train.py --model cnn         # 对照组: 小 CNN
python visualize.py                 # 可视化
```

参考结果（RTX 3060）：

| 模型 | 参数量 | 30 epoch 测试集准确率 | 用时 |
|---|---|---|---|
| ViT（patch 4，6 层，d=192） | 2.7M | 约 81% | 约 5 分钟 |
| 小 CNN（对照组） | 1.2M | 约 92% | 约 3 分钟 |

瞎猜的准确率是 10%。

训练脚本里有几个 ViT 常用的技巧，代码里都有注释：
- **数据增强**：随机平移 + 水平翻转，缓解数据少的问题
- **label smoothing**：正确类别的目标概率设为 0.9 而不是 1，防止模型过于自信
- **学习率预热 + 余弦退火**、**较大的 weight decay**
- **bfloat16 混合精度**：3060 上速度快约一倍

`visualize.py` 生成三张图：
- `vis_patches.png`：一张图被切成 64 个 patch 的样子
- `vis_pos_embed.png`：**学到的位置编码**。第 r 行第 c 列的小方块，表示 patch (r, c) 的位置向量和其他所有位置的相似度。看看亮的地方是不是在它附近
- `vis_attention.png`：CLS token 在最后一层关注图片的哪些区域

---

## 第四部分：动手实验

| # | 实验 | 命令 | 想一想 |
|---|---|---|---|
| 1 | ViT vs CNN | `python train.py --model cnn` | CNN 参数还不到 ViT 的一半，为什么准确率反而高 10 个百分点？ |
| 2 | 去掉数据增强 | `python train.py --no_aug` | 训练集和测试集准确率的差距变大了吗？ |
| 3 | patch 大小 | `--patch_size 8`（16 个 patch）对比默认的 4（64 个） | 序列变短，速度和准确率怎么变？ |
| 4 | 去掉位置编码 | 在 `forward` 里临时删掉 `+ self.pos_embed` | 准确率会掉多少？为什么不像项目 3 的倒序那样降到瞎猜？ |
| 5 | 看注意力 | 打开 `vis_attention.png` | CLS 关注的是物体本身还是背景？分类错的图片呢？ |

---

## 常见坑

1. **permute 的顺序**：`(B, C, h, P, w, P)` 要变成 `(B, h, w, C, P, P)`，即 `permute(0, 2, 4, 1, 3, 5)`。顺序错了不会报错，只是 patch 内容是乱的
2. **flatten 之后忘了 transpose**：Conv2d 输出 `(B, D, h, w)`，flatten 后是 `(B, D, N)`，要转成 `(B, N, D)`
3. **CLS 没有扩展到 batch**：`cls_token` 的形状是 `(1, 1, D)`，要 `expand(B, -1, -1)` 后才能和 `(B, N, D)` 拼接
4. **pos_embed 长度少了 1**：要给 CLS 也留一个位置，所以是 `N + 1`
5. **用了 `nn.Embedding` 当位置编码也可以**，但这里用 `nn.Parameter` 直接存一个 `(1, N+1, D)` 的张量，可以直接相加，更简洁

## 思考题

1. 32×32 的图片用 patch 4，序列长度是 65；如果用 patch 1（每个像素一个 token），序列长度是多少？attention 的计算量变成原来的几倍？这说明了为什么要切 patch。
2. 实验 4 去掉位置编码后，准确率为什么不会降到 10%？（提示：判断一张图是不是"飞机"，需要知道每个 patch 的位置吗？patch 内部的像素顺序还在吗？）
3. ViT 的位置编码只是一维编号 0~64，为什么学出来会有二维结构？
4. CLS token 和"所有 patch 输出取平均"相比，各有什么优缺点？
5. **联系 ACT**：ACT 为什么用 ResNet 提取图像特征，而不是直接像 ViT 这样切 patch？（提示：ACT 每个任务只有 50 条左右的演示数据，再想想第 1.5 节）
