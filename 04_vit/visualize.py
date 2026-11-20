"""
可视化训练好的 ViT（训练完再运行）

    python visualize.py

生成三张图:
    vis_patches.png    一张图被切成 64 个 patch 的样子
    vis_pos_embed.png  学到的位置编码: 每个 patch 的位置向量和其他位置有多相似
    vis_attention.png  CLS token 在最后一层关注图片的哪些区域
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from prepare_data import CLASSES
from vit import ViT, ViTConfig, patchify

HERE = Path(__file__).resolve().parent
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(HERE / "checkpoints" / "vit.pt", map_location=device)
cfg = ViTConfig(**ckpt["config"])
model = ViT(cfg).to(device).eval()
model.load_state_dict(ckpt["model"])
P, g = cfg.patch_size, cfg.img_size // cfg.patch_size   # patch 边长、每行 patch 数

d = np.load(HERE / "data" / "cifar10.npz")
torch.manual_seed(3)
idx = torch.randperm(len(d["x_test"]))[:8]
imgs = torch.from_numpy(d["x_test"][idx.numpy()]).to(device)
labels = d["y_test"][idx.numpy()]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
x = (imgs.float() / 255 - MEAN) / STD

# ----------------------------------------------------------------------------- 1. patch 切分
p = patchify(imgs[:1].float(), P)[0].reshape(-1, 3, P, P).permute(0, 2, 3, 1).cpu().numpy() / 255
fig, axes = plt.subplots(g, g, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(p[i])
    ax.axis("off")
fig.suptitle(f"{CLASSES[labels[0]]}: {g}x{g} = {g * g} patches of {P}x{P} pixels")
plt.savefig(HERE / "vis_patches.png", dpi=120, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------- 2. 位置编码相似度
pos = model.pos_embed[0, 1:].detach()                       # (N, D)，去掉 CLS 的那一个
sim = F.cosine_similarity(pos[:, None], pos[None, :], dim=-1).cpu()   # (N, N)
fig, axes = plt.subplots(g, g, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(sim[i].view(g, g), cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("cosine similarity between position embeddings\n"
             "(small square at row r, col c = similarity of patch (r,c) to every other patch)")
plt.savefig(HERE / "vis_pos_embed.png", dpi=120, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------- 3. CLS 的注意力
# 用 forward hook 截获最后一层 attention 的输出 (out, weights)
store = {}
model.blocks[-1].attn.register_forward_hook(lambda m, i, o: store.__setitem__("w", o[1]))
with torch.no_grad():
    pred = model(x).argmax(-1).cpu().numpy()
w = store["w"].mean(1)[:, 0, 1:].float()                    # 对头取平均，取 CLS 那一行，去掉 CLS 自己: (B, N)
w = w / w.amax(-1, keepdim=True)
heat = F.interpolate(w.view(-1, 1, g, g), size=cfg.img_size, mode="nearest")[:, 0].cpu().numpy()

fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
for j in range(8):
    img = imgs[j].permute(1, 2, 0).cpu().numpy()
    axes[0, j].imshow(img)
    ok = pred[j] == labels[j]
    axes[0, j].set_title(f"{CLASSES[labels[j]]}\npred: {CLASSES[pred[j]]}", fontsize=9, color="green" if ok else "red")
    axes[1, j].imshow(img)
    axes[1, j].imshow(heat[j], cmap="jet", alpha=0.5)
for a in axes.flat:
    a.axis("off")
fig.suptitle("top: test images (true / predicted)    bottom: where the CLS token attends in the last layer")
plt.savefig(HERE / "vis_attention.png", dpi=120, bbox_inches="tight")
plt.close()

print("已保存 vis_patches.png、vis_pos_embed.png、vis_attention.png")
