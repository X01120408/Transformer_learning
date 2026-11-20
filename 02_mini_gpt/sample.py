"""
用训练好的模型生成文本

    python sample.py
    python sample.py --prompt "ROMEO:" --n 500
    python sample.py --temperature 0.5      # 更保守、更通顺，但容易重复
    python sample.py --temperature 1.5      # 更随机、更"有创意"，但错字更多
    python sample.py --top_k 5              # 只在概率前 5 的字符里抽
"""
import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="\n")
ap.add_argument("--n", type=int, default=500, help="生成多少个字符")
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_k", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

ckpt_path = HERE / "checkpoints" / "ckpt.pt"
if not ckpt_path.exists():
    raise SystemExit("找不到 checkpoints/ckpt.pt，先运行 python train.py")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(ckpt_path, map_location=device)
chars = ckpt["chars"]
stoi = {ch: i for i, ch in enumerate(chars)}

model = GPT(GPTConfig(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

unknown = [c for c in args.prompt if c not in stoi]
if unknown:
    raise SystemExit(f"prompt 里有词表中没有的字符: {unknown}")

torch.manual_seed(args.seed)
idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
out = model.generate(idx, args.n, temperature=args.temperature, top_k=args.top_k)
print("".join(chars[i] for i in out[0].tolist()))
