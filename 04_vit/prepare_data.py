"""
下载 CIFAR-10 并转换成 numpy 格式（已写好，不需要修改）

    python prepare_data.py

CIFAR-10: 6 万张 32×32 的彩色小图，10 个类别（飞机、汽车、鸟、猫……），5 万张训练、1 万张测试。
从国内的 HuggingFace 镜像 hf-mirror.com 下载（共约 144MB），用 curl 断点续传。
下载完会转换成 data/cifar10.npz（uint8 数组，约 180MB），之后训练直接读取这个文件。
"""
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BASE = "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
FILES = {"train": ("train-00000-of-00001.parquet", 119705255),
         "test": ("test-00000-of-00001.parquet", 23940850)}
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def download(name, size, tries=20):
    out = DATA / name
    if out.exists() and out.stat().st_size == size:
        return out
    if shutil.which("curl") is None:
        raise SystemExit("没有找到 curl，请先安装: sudo apt install curl")
    for i in range(1, tries + 1):
        print(f"  下载 {name}（{size / 1e6:.0f} MB），第 {i} 次 ...")
        subprocess.run(["curl", "-L", "-C", "-", "--retry", "3", "--progress-bar", "-o", str(out), BASE + name])
        if out.exists() and out.stat().st_size >= size:
            return out
    raise SystemExit(f"下载失败，可以手动下载后放到 {out}:\n  {BASE + name}")


def to_numpy(path):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path).to_pydict()
    imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table["img"]])
    return imgs.transpose(0, 3, 1, 2), np.array(table["label"], dtype=np.int64)   # (N, 3, 32, 32), (N,)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    out = DATA / "cifar10.npz"
    if out.exists():
        print("data/cifar10.npz 已存在，跳过")
    else:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            raise SystemExit("需要 pyarrow: pip install pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple")
        arrays = {}
        for split, (name, size) in FILES.items():
            path = download(name, size)
            print(f"  解码 {split} 图片 ...")
            arrays[f"x_{split}"], arrays[f"y_{split}"] = to_numpy(path)
        np.savez(out, **arrays)
        for name, _ in FILES.values():
            (DATA / name).unlink()   # 删掉 parquet 原文件，只保留 npz

    d = np.load(out)
    print(f"\n训练集 {d['x_train'].shape}  测试集 {d['x_test'].shape}  类型 {d['x_train'].dtype}")
    print("每类训练样本数:", np.bincount(d["y_train"]).tolist())
    print("类别:", CLASSES)
