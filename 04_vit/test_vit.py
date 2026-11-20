"""
项目 4 测试脚本

用法:
    python test_vit.py        # 跑全部
    python test_vit.py 1      # 只跑 Step 1（可选 1~3）

你不需要修改这个文件。全部在 CPU 上运行，几秒钟跑完（不需要下载数据）。
"""
import math
import sys
import traceback

import torch
import torch.nn.functional as F

import vit as M

results = []


def check(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        results.append(True)
    except NotImplementedError as e:
        print(f"  ⏸  {name}   —— {e}")
        results.append(False)
    except AssertionError as e:
        print(f"  ❌ {name}\n       {e}")
        results.append(False)
    except Exception as e:
        print(f"  💥 {name}   —— 运行出错: {type(e).__name__}: {e}")
        for fr in traceback.extract_tb(e.__traceback__):
            if fr.filename.endswith(("/vit.py", "/seq2seq.py", "/model.py", "/attention.py")):
                print(f"       出错位置: {fr.filename.rsplit('/', 1)[-1]} 第 {fr.lineno} 行:  {fr.line}")
        results.append(False)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def small_cfg(**kw):
    base = dict(img_size=16, patch_size=4, in_chans=3, num_classes=10, d_model=32, n_layer=2, n_head=4)
    base.update(kw)
    return M.ViTConfig(**base)


# ----------------------------------------------------------------------------- Step 1
def step1():
    print("\n[Step 1] patchify")
    torch.manual_seed(0)
    img = torch.randn(2, 3, 32, 32)

    def shape():
        p = M.patchify(img, 4)
        assert p.shape == (2, 64, 48), f"输出形状 {tuple(p.shape)}，期望 (2, 64, 48) = (B, (32/4)², 3*4*4)"

    def first_patch():
        p = M.patchify(img, 4)
        exp = img[0, :, 0:4, 0:4].reshape(-1)
        assert torch.equal(p[0, 0], exp), "第 0 个 patch 应该是左上角 img[:, :, 0:4, 0:4]，并按 (C, P, P) 顺序拉平"

    def row_major():
        p = M.patchify(img, 4)
        assert torch.equal(p[1, 1], img[1, :, 0:4, 4:8].reshape(-1)), \
            "第 1 个 patch 应该在第 0 个的右边（行优先）: img[:, :, 0:4, 4:8]"
        assert torch.equal(p[1, 8], img[1, :, 4:8, 0:4].reshape(-1)), \
            "第 8 个 patch 应该是第二行的第一个: img[:, :, 4:8, 0:4]"

    def match_unfold():
        p = M.patchify(img, 8)
        ref = F.unfold(img, kernel_size=8, stride=8).transpose(1, 2)
        assert p.shape == ref.shape and torch.equal(p, ref), "和 PyTorch 官方 F.unfold 的结果不一致"

    check("输出形状正确", shape)
    check("第 0 个 patch 内容正确", first_patch)
    check("patch 按行优先排列", row_major)
    check("与 F.unfold 一致（P=8）", match_unfold)


# ----------------------------------------------------------------------------- Step 2
def step2():
    print("\n[Step 2] PatchEmbedding")
    cfg = small_cfg()
    C, P, D = cfg.in_chans, cfg.patch_size, cfg.d_model

    def shape():
        y = M.PatchEmbedding(cfg)(torch.randn(2, C, 16, 16))
        assert y.shape == (2, cfg.n_patches, D), f"输出形状 {tuple(y.shape)}，期望 {(2, cfg.n_patches, D)}"

    def params():
        n, exp = n_params(M.PatchEmbedding(cfg)), C * P * P * D + D
        assert n == exp, f"参数量 {n}，期望 {exp}（等价于一个 Linear(C*P*P, D)）"

    def equals_linear():
        torch.manual_seed(0)
        pe = M.PatchEmbedding(cfg)
        img = torch.randn(2, C, 16, 16)
        conv = [m for m in pe.modules() if isinstance(m, torch.nn.Conv2d)]
        assert conv, "PatchEmbedding 里应该有一个 nn.Conv2d"
        W, b = conv[0].weight.reshape(D, -1), conv[0].bias
        ref = M.patchify(img, P) @ W.T + b
        err = (pe(img) - ref).abs().max().item()
        assert err < 1e-5, (f"Conv2d 的结果和 patchify + Linear 不一致（误差 {err:.2e}）。"
                            f"检查 kernel_size / stride 是否都等于 patch_size，以及 flatten / transpose 的维度")

    check("输出形状正确", shape)
    check("参数量正确", params)
    check("Conv2d 等价于 patchify + Linear", equals_linear)


# ----------------------------------------------------------------------------- Step 3
def step3():
    print("\n[Step 3] ViT")
    cfg = small_cfg()
    C, P, D, N, L, K = cfg.in_chans, cfg.patch_size, cfg.d_model, cfg.n_patches, cfg.n_layer, cfg.num_classes

    def shape():
        y = M.ViT(cfg)(torch.randn(3, C, 16, 16))
        assert y.shape == (3, K), f"logits 形状 {tuple(y.shape)}，期望 {(3, K)}"

    def params():
        n = n_params(M.ViT(cfg))
        exp = (C * P * P * D + D) + D + (N + 1) * D + L * (12 * D * D + 13 * D) + 2 * D + (D * K + K)
        assert n == exp, (f"参数量 {n}，期望 {exp} = patch_embed + cls_token + pos_embed(N+1 个) "
                          f"+ {L} 个 EncoderLayer + LayerNorm + head")

    def init_loss():
        torch.manual_seed(0)
        logits = M.ViT(cfg)(torch.randn(64, C, 16, 16))
        loss = F.cross_entropy(logits, torch.randint(0, K, (64,))).item()
        assert abs(loss - math.log(K)) < 0.3, f"初始 loss {loss:.3f}，应接近 ln({K}) = {math.log(K):.3f}"

    def uses_position():
        torch.manual_seed(0)
        m = M.ViT(cfg).eval()
        img = torch.randn(1, C, 16, 16)
        swapped = img.clone()
        swapped[..., 0:4, 0:4], swapped[..., 12:16, 12:16] = img[..., 12:16, 12:16], img[..., 0:4, 0:4]
        assert (m(img) - m(swapped)).abs().max().item() > 1e-6, \
            "交换两个 patch 的位置后输出没变 —— 位置编码没有加上"

    def cls_output():
        torch.manual_seed(0)
        m = M.ViT(cfg).eval()
        feats = {}
        m.ln.register_forward_hook(lambda mod, i, o: feats.setdefault("x", o))
        logits = m(torch.randn(2, C, 16, 16))
        x = feats["x"]
        assert x.shape[1] == N + 1, f"进入最后 LayerNorm 的序列长度 {x.shape[1]}，期望 {N + 1}（{N} 个 patch + 1 个 CLS）"
        assert (m.head(x[:, 0]) - logits).abs().max().item() < 1e-5, "分类头应该只用第 0 个位置（CLS）的输出"

    def overfit():
        torch.manual_seed(0)
        m = M.ViT(cfg)
        img, y = torch.randn(16, C, 16, 16), torch.randint(0, K, (16,))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        for _ in range(150):
            loss = F.cross_entropy(m(img), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < 0.05, f"在 16 张图上训练 150 步后 loss = {loss.item():.3f}（应 < 0.05），说明有 bug"

    check("logits 形状正确", shape)
    check("参数量正确", params)
    check("初始 loss ≈ ln(10)", init_loss)
    check("使用了位置编码", uses_position)
    check("序列长度 N+1，分类用 CLS 位置", cls_output)
    check("能在一个小 batch 上过拟合", overfit)


if __name__ == "__main__":
    steps = {"1": step1, "2": step2, "3": step3}
    chosen = sys.argv[1:] or list(steps)
    for s in chosen:
        if s not in steps:
            sys.exit(f"未知的 step: {s}，可选 1~3")
        steps[s]()
    n_pass, n_all = sum(results), len(results)
    print(f"\n通过 {n_pass}/{n_all}")
    if n_pass == n_all and not sys.argv[1:]:
        print("🎉 全部通过！下一步: python prepare_data.py，然后 python train.py")
