"""
项目 3 测试脚本

用法:
    python test_seq2seq.py        # 跑全部
    python test_seq2seq.py 1      # 只跑 Step 1（可选 1~5，也可以 2 3）

你不需要修改这个文件。全部在 CPU 上运行，十几秒跑完。
"""
import math
import sys
import traceback

import torch

import seq2seq as M
from data import VOCAB_SIZE, make_batch

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
            if fr.filename.endswith(("/seq2seq.py", "/model.py", "/attention.py")):
                print(f"       出错位置: {fr.filename.rsplit('/', 1)[-1]} 第 {fr.lineno} 行:  {fr.line}")
        results.append(False)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def cfg(**kw):
    base = dict(vocab_size=VOCAB_SIZE, src_len=10, tgt_len=10, n_layer=2, n_head=4, d_model=32, dropout=0.0)
    base.update(kw)
    return M.Seq2SeqConfig(**base)


def diff(a, b):
    return (a - b).abs().max().item()


def zero_all(m):
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()


def overfit(model, task, steps=300, lr=3e-3):
    torch.manual_seed(0)
    src, tgt = make_batch(task, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        _, loss = model(src, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()


# ----------------------------------------------------------------------------- Step 1
def step1():
    print("\n[Step 1] EncoderLayer")
    c = cfg()
    D = c.d_model

    def shape():
        y = M.EncoderLayer(c)(torch.randn(2, 7, D))
        assert y.shape == (2, 7, D), f"输出形状 {tuple(y.shape)}，期望 {(2, 7, D)}"

    def params():
        n, exp = n_params(M.EncoderLayer(c)), 12 * D * D + 13 * D
        assert n == exp, f"参数量 {n}，期望 {exp}（和项目 2 的 Block 一样）"

    def bidirectional():
        torch.manual_seed(0)
        layer = M.EncoderLayer(c).eval()
        x = torch.randn(1, 6, D)
        x2 = x.clone()
        x2[:, -1] = torch.randn(D)
        assert diff(layer(x)[:, 0], layer(x2)[:, 0]) > 1e-6, \
            "修改最后一个位置，第一个位置的输出没变 —— Encoder 不应该用因果掩码，每个位置要能看到所有位置"

    def residual():
        layer = M.EncoderLayer(c).eval()
        zero_all(layer)
        x = torch.randn(2, 5, D)
        assert diff(layer(x), x) < 1e-6, "所有参数为 0 时应原样输出 x（检查残差连接）"

    check("输出形状正确", shape)
    check("参数量正确", params)
    check("双向: 每个位置能看到所有位置", bidirectional)
    check("残差连接正确", residual)


# ----------------------------------------------------------------------------- Step 2
def step2():
    print("\n[Step 2] DecoderLayer")
    c = cfg()
    D, T, S = c.d_model, 5, 9

    def shape():
        y = M.DecoderLayer(c)(torch.randn(2, T, D), torch.randn(2, S, D))
        assert y.shape == (2, T, D), f"输出形状 {tuple(y.shape)}，期望 {(2, T, D)}（输出长度跟着 y 走，不跟 memory）"

    def params():
        n, exp = n_params(M.DecoderLayer(c)), 16 * D * D + 19 * D
        assert n == exp, f"参数量 {n}，期望 {exp} = 2 个 Attention(8D²+8D) + FeedForward(8D²+5D) + 3 个 LayerNorm(6D)"

    def causal():
        torch.manual_seed(0)
        layer = M.DecoderLayer(c).eval()
        y, mem = torch.randn(1, T, D), torch.randn(1, S, D)
        y2 = y.clone()
        y2[:, 3:] = torch.randn(T - 3, D)
        assert diff(layer(y, mem)[:, :3], layer(y2, mem)[:, :3]) < 1e-5, \
            "causal=True 时，修改后面的位置不应影响前面位置的输出（self-attention 要传因果掩码）"
        assert diff(layer(y, mem, causal=False)[:, :3], layer(y2, mem, causal=False)[:, :3]) > 1e-6, \
            "causal=False 时，所有位置应该能互相看到（mask 应为 None）"

    def uses_memory():
        torch.manual_seed(0)
        layer = M.DecoderLayer(c).eval()
        y = torch.randn(1, T, D)
        assert diff(layer(y, torch.randn(1, S, D)), layer(y, torch.randn(1, S, D))) > 1e-6, \
            "换一个 memory 输出却没变 —— cross-attention 的 key/value 应该来自 memory"

    def cross_weights():
        torch.manual_seed(0)
        layer = M.DecoderLayer(c).eval()
        layer(torch.randn(2, T, D), torch.randn(2, S, D))
        w = getattr(layer, "cross_weights", None)
        assert w is not None, "没有保存 self.cross_weights"
        assert w.shape == (2, c.n_head, T, S), f"cross_weights 形状 {tuple(w.shape)}，期望 {(2, c.n_head, T, S)}"
        assert diff(w.sum(-1), torch.ones(2, c.n_head, T)) < 1e-5, "cross_weights 每行之和应为 1"

    def residual():
        layer = M.DecoderLayer(c).eval()
        zero_all(layer)
        y = torch.randn(2, T, D)
        assert diff(layer(y, torch.randn(2, S, D)), y) < 1e-6, "所有参数为 0 时应原样输出 y（检查 3 个残差连接）"

    check("输出形状正确（T ≠ S）", shape)
    check("参数量正确", params)
    check("causal=True 看不到未来，causal=False 能看到全部", causal)
    check("cross-attention 使用了 memory", uses_memory)
    check("保存了 cross_weights", cross_weights)
    check("残差连接正确", residual)


# ----------------------------------------------------------------------------- Step 3
def step3():
    print("\n[Step 3] Seq2Seq encode / decode / forward")
    c = cfg(src_len=9, tgt_len=5)
    V, D, L = c.vocab_size, c.d_model, c.n_layer

    def params():
        n = n_params(M.Seq2Seq(c))
        exp = V * D + 9 * D + 5 * D + L * (12 * D * D + 13 * D) + L * (16 * D * D + 19 * D) + 4 * D + D * V
        assert n == exp, f"参数量 {n}，期望 {exp} = tok_emb + src_pos + tgt_pos + encoder + decoder + 2 个 LN + head"

    def shapes():
        m = M.Seq2Seq(c)
        src, tgt = make_batch("add", 3)
        mem = m.encode(src)
        assert mem.shape == (3, 9, D), f"encode 输出形状 {tuple(mem.shape)}，期望 {(3, 9, D)}"
        ret = m(src, tgt)
        assert isinstance(ret, tuple) and len(ret) == 2, "forward 应返回 (logits, loss)"
        assert ret[0].shape == (3, 5, V), f"logits 形状 {tuple(ret[0].shape)}，期望 {(3, 5, V)}"
        assert ret[1].dim() == 0, "loss 应该是标量"

    def init_loss():
        torch.manual_seed(0)
        _, loss = M.Seq2Seq(c)(*make_batch("add", 32))
        assert abs(loss.item() - math.log(V)) < 0.3, f"初始 loss {loss.item():.3f}，应接近 ln({V}) = {math.log(V):.3f}"

    def teacher_forcing():
        torch.manual_seed(0)
        m = M.Seq2Seq(c).eval()
        src, tgt = make_batch("add", 2)
        t = 2
        tgt2 = tgt.clone()
        tgt2[:, t:] = (tgt2[:, t:] + 3) % 10
        l1, l2 = m(src, tgt)[0], m(src, tgt2)[0]
        assert diff(l1[:, :t + 1], l2[:, :t + 1]) < 1e-5, (
            f"修改了 tgt[:, {t}:]，位置 0~{t} 的 logits 却变了。预测 tgt[{t}] 时不能看到 tgt[{t}] 本身 —— "
            f"检查 decoder 输入是否右移了一位（开头补 <s>），以及是否传了因果掩码")
        assert diff(l1[:, t + 1:], l2[:, t + 1:]) > 1e-6, "后面位置的 logits 应该随 tgt 改变"

    def uses_src():
        torch.manual_seed(0)
        m = M.Seq2Seq(c).eval()
        src, tgt = make_batch("add", 2)
        assert diff(m(src, tgt)[0], m(make_batch("add", 2)[0], tgt)[0]) > 1e-6, "换了 src，logits 却没变"

    def overfit_test():
        loss = overfit(M.Seq2Seq(cfg()), "reverse")
        assert loss < 0.1, f"在一个小 batch 上训练 300 步后 loss = {loss:.3f}（应 < 0.1），说明有 bug"

    check("参数量正确", params)
    check("encode / forward 输出形状正确", shapes)
    check("初始 loss ≈ ln(vocab_size)", init_loss)
    check("Teacher Forcing 正确（右移一位 + 因果掩码）", teacher_forcing)
    check("输出依赖输入 src", uses_src)
    check("能在一个小 batch 上过拟合", overfit_test)


# ----------------------------------------------------------------------------- Step 4
def step4():
    print("\n[Step 4] Seq2Seq.generate")
    c = cfg(src_len=9, tgt_len=5)

    def shape():
        torch.manual_seed(0)
        out = M.Seq2Seq(c).eval().generate(make_batch("add", 3)[0])
        assert out.shape == (3, 5), f"输出形状 {tuple(out.shape)}，期望 (3, 5)（不含开头的 <s>）"

    def greedy_consistent():
        torch.manual_seed(0)
        m = M.Seq2Seq(c).eval()
        src = make_batch("add", 4)[0]
        out = m.generate(src)
        logits, _ = m(src, out)
        assert torch.equal(logits.argmax(-1), out), (
            "自洽性检查失败: 把 generate 的输出当作答案喂回 forward，每个位置的 argmax 应该正好是 generate 选的字符。"
            "检查是否只取了最后一个位置、是否用 argmax、是否去掉了开头的 <s>")

    def learns_reverse():
        torch.manual_seed(0)
        m = M.Seq2Seq(cfg())
        overfit(m, "reverse")
        torch.manual_seed(0)   # 重新生成 overfit 里用过的同一个 batch
        src, tgt = make_batch("reverse", 8)
        acc = (m.eval().generate(src) == tgt).float().mean().item()
        assert acc > 0.95, f"在训练过的 batch 上，generate 的字符准确率只有 {acc:.0%}"

    check("输出形状正确", shape)
    check("贪心解码自洽", greedy_consistent)
    check("过拟合后能正确生成", learns_reverse)


# ----------------------------------------------------------------------------- Step 5
def step5():
    print("\n[Step 5] ParallelSeq2Seq（ACT 风格）")
    c = cfg(src_len=9, tgt_len=5)
    D = c.d_model

    def params():
        n, exp = n_params(M.ParallelSeq2Seq(c)), n_params(M.Seq2Seq(c)) + 5 * D
        assert n == exp, f"参数量 {n}，期望 {exp}（比 Seq2Seq 多一个 tgt_len × D 的 query_embed）"

    def shapes():
        m = M.ParallelSeq2Seq(c)
        src, tgt = make_batch("add", 3)
        logits, loss = m(src, tgt)
        assert logits.shape == (3, 5, c.vocab_size), f"logits 形状 {tuple(logits.shape)}"
        assert loss.dim() == 0, "loss 应该是标量"
        assert m.eval().generate(src).shape == (3, 5), "generate 输出形状应为 (B, tgt_len)"

    def no_tgt_input():
        torch.manual_seed(0)
        m = M.ParallelSeq2Seq(c).eval()
        src, tgt = make_batch("add", 2)
        assert diff(m(src, tgt)[0], m(src, (tgt + 1) % 10)[0]) < 1e-6, \
            "logits 随 tgt 变化了 —— 并行解码时 tgt 只用来算 loss，decoder 的输入应该是 query_embed"

    def not_causal():
        torch.manual_seed(0)
        m = M.ParallelSeq2Seq(c).eval()
        mem = m.encode(make_batch("add", 1)[0])
        base = m.decode_parallel(mem)
        with torch.no_grad():
            m.query_embed.weight[-1] += torch.randn(D)   # 随机扰动（加常数会被 LayerNorm 抵消）
        assert diff(base[:, 0], m.decode_parallel(mem)[:, 0]) > 1e-6, \
            "修改最后一个查询，第一个位置的输出没变 —— 并行解码应使用 causal=False"

    def generate_is_argmax():
        torch.manual_seed(0)
        m = M.ParallelSeq2Seq(c).eval()
        src, tgt = make_batch("add", 4)
        assert torch.equal(m.generate(src), m(src, tgt)[0].argmax(-1)), "generate 应等于 logits 在最后一维的 argmax"

    def overfit_test():
        loss = overfit(M.ParallelSeq2Seq(cfg()), "reverse")
        assert loss < 0.1, f"在一个小 batch 上训练 300 步后 loss = {loss:.3f}（应 < 0.1）"

    check("参数量正确", params)
    check("输出形状正确", shapes)
    check("decoder 不使用 tgt 作为输入", no_tgt_input)
    check("查询之间能互相看到（无因果掩码）", not_causal)
    check("generate = argmax，一次输出全部", generate_is_argmax)
    check("能在一个小 batch 上过拟合", overfit_test)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    steps = {"1": step1, "2": step2, "3": step3, "4": step4, "5": step5}
    chosen = sys.argv[1:] or list(steps)
    for s in chosen:
        if s not in steps:
            sys.exit(f"未知的 step: {s}，可选 1~5")
        steps[s]()

    n_pass, n_all = sum(results), len(results)
    print(f"\n通过 {n_pass}/{n_all}")
    if n_pass == n_all and not sys.argv[1:]:
        print("🎉 全部通过！下一步: python train.py --task reverse")
