"""
项目 3 的数据：程序随机生成，不需要下载（已写好，不需要修改）

三个任务，输入和输出都是数字串:
    reverse  倒序:  3 1 4 1 5 9 2 6 5 3   ->  3 5 6 2 9 5 1 4 1 3
    sort     排序:  3 1 4 1 5 9 2 6 5 3   ->  1 1 2 3 3 4 5 5 6 9
    add      加法:  0 4 7 3 + 0 9 1 8     ->  0 1 3 9 1        (473 + 918 = 1391，固定 5 位)

    python data.py          # 打印几个样例看看
"""
import torch

VOCAB = list("0123456789+") + ["<s>"]   # <s> = 解码开始符（BOS），自回归解码时作为第一个输入
BOS = VOCAB.index("<s>")
PLUS = VOCAB.index("+")
VOCAB_SIZE = len(VOCAB)                  # 12

# 每个任务的输入长度 src_len、输出长度 tgt_len（全部固定长度，不需要 padding）
TASKS = {
    "reverse": dict(src_len=10, tgt_len=10),
    "sort":    dict(src_len=10, tgt_len=10),
    "add":     dict(src_len=9,  tgt_len=5),
}


def make_batch(task, batch_size, device="cpu"):
    """返回 src (B, src_len), tgt (B, tgt_len)，都是 long 型的 token 编号"""
    B = batch_size
    if task in ("reverse", "sort"):
        src = torch.randint(0, 10, (B, TASKS[task]["src_len"]))
        tgt = src.flip(1) if task == "reverse" else src.sort(dim=1).values
    elif task == "add":
        a, b = torch.randint(0, 10000, (B,)), torch.randint(0, 10000, (B,))
        src = torch.cat([digits(a, 4), torch.full((B, 1), PLUS), digits(b, 4)], dim=1)
        tgt = digits(a + b, 5)
    else:
        raise ValueError(f"未知任务 {task}，可选 {list(TASKS)}")
    return src.to(device), tgt.to(device)


def digits(x, n):
    """整数 -> n 位数字 token（高位在前，不足补 0）。例: digits(473, 4) = [0, 4, 7, 3]"""
    p = 10 ** torch.arange(n - 1, -1, -1)
    return (x[:, None] // p) % 10


def to_str(ids):
    return " ".join(VOCAB[i] for i in ids.tolist())


if __name__ == "__main__":
    torch.manual_seed(0)
    for task in TASKS:
        src, tgt = make_batch(task, 2)
        print(f"[{task}]")
        for s, t in zip(src, tgt):
            print(f"  {to_str(s):<22} ->  {to_str(t)}")
