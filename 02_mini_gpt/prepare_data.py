"""
下载训练数据 tiny Shakespeare（莎士比亚剧本节选，约 1MB，共 111 万个字符，但字符种类只有 65 种）

    python prepare_data.py                   # 下载到 data/input.txt
    python prepare_data.py --from 我的.txt    # 改用你自己的文本（比如唐诗、小说，需 UTF-8 编码）

GitHub 在国内偶尔断线，脚本会用 curl 自动断点续传重试。
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SIZE = 1115394
OUT = Path(__file__).resolve().parent / "data" / "input.txt"


def download(url, out, size, tries=10):
    """用 curl 断点续传下载（国内网络下 curl 连 GitHub 比 Python 自带的 urllib 稳定）"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    if shutil.which("curl") is None:
        raise SystemExit("没有找到 curl，请先安装: sudo apt install curl")
    for i in range(1, tries + 1):
        subprocess.run(["curl", "-sL", "-C", "-", "--retry", "3", "--max-time", "120", "-o", str(tmp), url])
        have = tmp.stat().st_size if tmp.exists() else 0
        if have >= size:
            break
        print(f"  第 {i} 次下载中断，已下载 {have / 1e6:.2f} MB，重试...")
        time.sleep(1)
    if not tmp.exists() or tmp.stat().st_size < size:
        raise SystemExit("下载失败。可以手动在浏览器打开下面的网址，另存为 02_mini_gpt/data/input.txt:\n  " + url)
    tmp.rename(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", help="使用自己的 UTF-8 文本文件")
    args = ap.parse_args()

    if args.src:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.src, OUT)
    elif OUT.exists() and OUT.stat().st_size == SIZE:
        print("data/input.txt 已存在，跳过下载")
    else:
        print("下载 tiny Shakespeare ...")
        download(URL, OUT, SIZE)

    text = OUT.read_text(encoding="utf-8")
    chars = sorted(set(text))
    print(f"\n已保存 {OUT}")
    print(f"总字符数: {len(text):,}    不同字符数（词表大小）: {len(chars)}")
    print("词表:", repr("".join(chars)))
    print("\n开头 300 个字符:\n" + "-" * 40 + "\n" + text[:300])
