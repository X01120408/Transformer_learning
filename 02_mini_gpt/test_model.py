"""
项目 2 测试脚本

用法:
    python test_model.py        # 跑全部
    python test_model.py 1      # 只跑 Step 1（也可以 2、3、4，或者 2 3）

你不需要修改这个文件。全部在 CPU 上运行，几秒钟就跑完。
"""
import math
import sys
import traceback

import torch

import model as M

results = []


# ----------------------------------------------------------------------------- 工具
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
            if fr.filename.endswith(("/model.py", "/attention.py")):
                fname = fr.filename.rsplit("/", 1)[-1]
                print(f"       出错位置: {fname} 第 {fr.lineno} 行:  {fr.line}")
        results.append(False)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def small_cfg(**kw):
    base = dict(vocab_size=11, block_size=16, n_layer=2, n_head=4, d_model=32, dropout=0.0)
    base.update(kw)
    return M.GPTConfig(**base)


def assert_causal(f, x, what):
    """改动第 t 个位置之后的输入，前 t 个位置的输出不应该变化"""
    T = x.size(1)
    t = T // 2
    y1 = f(x)
    x2 = x.clone()
    if x2.dtype == torch.long:
        x2[:, t:] = (x2[:, t:] + 1) % 11
    else:
        x2[:, t:] = torch.randn_like(x2[:, t:])
    y2 = f(x2)
    err = (y1[:, :t] - y2[:, :t]).abs().max().item()
    assert err < 1e-5, (f"{what}: 修改了第 {t} 个位置之后的输入，前 {t} 个位置的输出却变了（差 {err:.2e}）。"
                        f"说明模型偷看了未来 —— 检查因果 mask 是否传进了 attention")
    assert (y1[:, t:] - y2[:, t:]).abs().max().item() > 1e-6, f"{what}: 修改输入后，后面位置的输出完全没变，不正常"


# ----------------------------------------------------------------------------- Step 1
def step1():
    print("\n[Step 1] FeedForward")
    D = 32

    def shape():
        ffn = M.FeedForward(D)
        y = ffn(torch.randn(2, 5, D))
        assert y.shape == (2, 5, D), f"输出形状 {tuple(y.shape)}，期望 {(2, 5, D)}"

    def params():
        n, exp = n_params(M.FeedForward(D)), 8 * D * D + 5 * D
        assert n == exp, f"参数量 {n}，期望 {exp}（D->4D 和 4D->D 两个带 bias 的 Linear）"

    def per_position():
        torch.manual_seed(0)
        ffn = M.FeedForward(D).eval()
        x = torch.randn(2, 5, D)
        y = ffn(x)
        y3 = ffn(x[:, 3:4])
        assert (y[:, 3:4] - y3).abs().max().item() < 1e-5, "FeedForward 应该对每个位置独立计算，和其他位置无关"

    def nonlinear():
        torch.manual_seed(0)
        ffn = M.FeedForward(D).eval()
        a, b = torch.randn(1, 1, D), torch.randn(1, 1, D)
        lin = (ffn(a + b) - ffn(a) - ffn(b) + ffn(torch.zeros(1, 1, D))).abs().max().item()
        assert lin > 1e-4, "FeedForward 是线性的 —— 两个 Linear 中间少了激活函数 GELU"

    check("输出形状正确", shape)
    check("参数量正确", params)
    check("每个位置独立计算", per_position)
    check("包含非线性激活", nonlinear)


# ----------------------------------------------------------------------------- Step 2
def step2():
    print("\n[Step 2] Block")
    cfg = small_cfg()
    D = cfg.d_model

    def shape():
        y = M.Block(cfg)(torch.randn(2, 7, D))
        assert y.shape == (2, 7, D), f"输出形状 {tuple(y.shape)}，期望 {(2, 7, D)}"

    def params():
        n, exp = n_params(M.Block(cfg)), 12 * D * D + 13 * D
        assert n == exp, (f"参数量 {n}，期望 {exp} = Attention(4D²+4D) + FeedForward(8D²+5D) + 2 个 LayerNorm(4D)")

    def causal():
        torch.manual_seed(0)
        blk = M.Block(cfg).eval()
        assert_causal(blk, torch.randn(2, 8, D), "Block")

    def residual():
        torch.manual_seed(0)
        blk = M.Block(cfg).eval()
        with torch.no_grad():
            for p in blk.parameters():   # 所有参数清零: 子层输出全为 0，只剩残差通路
                p.zero_()
        x = torch.randn(2, 5, D)
        err = (blk(x) - x).abs().max().item()
        assert err < 1e-6, "所有权重为 0 时 Block 应该原样输出 x（检查残差连接 x = x + ...）"

    def gpu_mask():
        if not torch.cuda.is_available():
            return
        blk = M.Block(cfg).cuda()
        blk(torch.randn(2, 5, D, device="cuda"))

    check("输出形状正确", shape)
    check("参数量正确", params)
    check("因果性: 看不到未来", causal)
    check("残差连接正确", residual)
    check("能在 GPU 上运行（mask 的 device 正确）", gpu_mask)


# ----------------------------------------------------------------------------- Step 3
def step3():
    print("\n[Step 3] GPT forward")
    cfg = small_cfg()
    V, D, L, C = cfg.vocab_size, cfg.d_model, cfg.n_layer, cfg.block_size

    def shape_and_none_loss():
        g = M.GPT(cfg)
        ret = g(torch.randint(0, V, (3, 10)))
        assert isinstance(ret, tuple) and len(ret) == 2, "forward 应该返回 (logits, loss)"
        logits, loss = ret
        assert logits.shape == (3, 10, V), f"logits 形状 {tuple(logits.shape)}，期望 {(3, 10, V)}"
        assert loss is None, "没传 targets 时 loss 应该是 None"

    def params():
        n = n_params(M.GPT(cfg))
        exp = V * D + C * D + L * (12 * D * D + 13 * D) + 2 * D + D * V
        assert n == exp, (f"参数量 {n}，期望 {exp} = tok_emb + pos_emb + {L} 个 Block + ln_f + head（head 无 bias）")

    def init_loss():
        torch.manual_seed(0)
        g = M.GPT(cfg)
        idx = torch.randint(0, V, (8, 16))
        _, loss = g(idx, torch.randint(0, V, (8, 16)))
        assert loss is not None and loss.dim() == 0, "传了 targets 时 loss 应该是一个标量"
        exp = math.log(V)
        assert abs(loss.item() - exp) < 0.3, (
            f"刚初始化的模型 loss = {loss.item():.3f}，应该接近 ln({V}) = {exp:.3f}（随机猜的水平）")

    def causal():
        torch.manual_seed(0)
        g = M.GPT(cfg).eval()
        assert_causal(lambda i: g(i)[0], torch.randint(0, V, (2, 12)), "GPT")

    def overfit_one_batch():
        torch.manual_seed(0)
        g = M.GPT(cfg)
        x = torch.randint(0, V, (4, 16))
        y = torch.roll(x, -1, dims=1)
        opt = torch.optim.AdamW(g.parameters(), lr=3e-3)
        for _ in range(150):
            _, loss = g(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < 0.3, (f"在同一个小 batch 上训练 150 步后 loss 还有 {loss.item():.3f}（应 < 0.3）。"
                                   f"模型连背下一个 batch 都做不到，说明有 bug")

    check("logits 形状正确，无 targets 时 loss 为 None", shape_and_none_loss)
    check("参数量正确", params)
    check("初始 loss ≈ ln(vocab_size)", init_loss)
    check("整个模型看不到未来", causal)
    check("能在一个小 batch 上过拟合（loss → 0）", overfit_one_batch)


# ----------------------------------------------------------------------------- Step 4
def step4():
    print("\n[Step 4] generate")
    cfg = small_cfg()
    V = cfg.vocab_size

    def shape_and_prefix():
        torch.manual_seed(0)
        g = M.GPT(cfg).eval()
        idx = torch.randint(0, V, (2, 3))
        out = g.generate(idx, 5)
        assert out.shape == (2, 8), f"输出形状 {tuple(out.shape)}，期望 (2, 8)"
        assert torch.equal(out[:, :3], idx), "输出的开头应该保留原来的 prompt"
        assert out.min() >= 0 and out.max() < V, "生成的编号超出了词表范围"

    def longer_than_block():
        torch.manual_seed(0)
        g = M.GPT(cfg).eval()
        out = g.generate(torch.zeros(1, 1, dtype=torch.long), cfg.block_size + 10)
        assert out.shape == (1, cfg.block_size + 11), f"输出形状 {tuple(out.shape)}"

    def top1_is_greedy():
        torch.manual_seed(0)
        g = M.GPT(cfg).eval()
        idx = torch.randint(0, V, (2, 4))
        out = g.generate(idx, 6, top_k=1)
        ref = idx
        for _ in range(6):
            logits, _ = g(ref[:, -cfg.block_size:])
            ref = torch.cat([ref, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        assert torch.equal(out, ref), "top_k=1 时应该每次都选概率最大的字符（检查 top_k 和是否只用了最后一个位置）"

    check("输出形状正确、保留 prompt", shape_and_prefix)
    check("生成长度超过 block_size 也不报错（裁剪上下文）", longer_than_block)
    check("top_k=1 等价于贪心解码", top1_is_greedy)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    steps = {"1": step1, "2": step2, "3": step3, "4": step4}
    chosen = sys.argv[1:] or list(steps)
    for s in chosen:
        if s not in steps:
            sys.exit(f"未知的 step: {s}，可选 1 / 2 / 3 / 4")
        steps[s]()

    n_pass, n_all = sum(results), len(results)
    print(f"\n通过 {n_pass}/{n_all}")
    if n_pass == n_all and not sys.argv[1:]:
        print("🎉 全部通过！下一步: python prepare_data.py，然后 python train.py")
