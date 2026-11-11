"""
项目 1 测试脚本 —— 把你的实现和 PyTorch 官方实现逐项对比

用法:
    python test_attention.py        # 跑全部
    python test_attention.py 1      # 只跑 Step 1（也可以 2、3，或者 1 2）

你不需要修改这个文件。
"""
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

import attention as A

ATOL = 1e-5
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
            if fr.filename.endswith("/attention.py") or fr.filename == "attention.py":
                print(f"       出错位置: attention.py 第 {fr.lineno} 行:  {fr.line}")
        results.append(False)


def assert_close(actual, expected, what):
    assert isinstance(actual, torch.Tensor), f"{what} 应该是 Tensor，得到 {type(actual)}"
    assert actual.shape == expected.shape, (
        f"{what} 形状不对: 得到 {tuple(actual.shape)}，期望 {tuple(expected.shape)}")
    err = (actual.float() - expected.float()).abs().max().item()
    assert err < ATOL, f"{what} 数值不对: 最大误差 {err:.2e}（允许 {ATOL:.0e}）"


def unpack(ret, fname):
    assert isinstance(ret, tuple) and len(ret) == 2, f"{fname} 应该返回 (out, weights) 两个值"
    return ret


# ----------------------------------------------------------------------------- Step 1
def step1():
    print("\n[Step 1] scaled_dot_product_attention")
    torch.manual_seed(0)
    B, H, L, S, dk, dv = 2, 3, 5, 7, 8, 6
    q, k, v = torch.randn(B, H, L, dk), torch.randn(B, H, S, dk), torch.randn(B, H, S, dv)
    sdpa = lambda *a: unpack(A.scaled_dot_product_attention(*a), "scaled_dot_product_attention")

    def shapes():
        out, w = sdpa(q, k, v)
        assert out.shape == (B, H, L, dv), f"out 形状 {tuple(out.shape)}，期望 {(B, H, L, dv)}"
        assert w.shape == (B, H, L, S), f"weights 形状 {tuple(w.shape)}，期望 {(B, H, L, S)}"

    def rows_sum_to_one():
        _, w = sdpa(q, k, v)
        assert_close(w.sum(-1), torch.ones(B, H, L), "weights 每一行之和（应为 1，检查 softmax 的 dim）")

    def match_no_mask():
        out, _ = sdpa(q, k, v)
        assert_close(out, F.scaled_dot_product_attention(q, k, v),
                     "out（检查是否除以了 sqrt(d_k)）")

    def match_with_mask():
        g = torch.Generator().manual_seed(1)
        mask = torch.rand(L, S, generator=g) > 0.4
        mask[:, 0] = True  # 保证每行至少有一个可见位置
        out, w = sdpa(q, k, v, mask)
        assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask), "out（带 mask）")
        leaked = (w * (~mask)).abs().max().item()
        assert leaked == 0, f"被屏蔽（mask=False）的位置权重应该是 0，实际最大为 {leaked:.3g}"

    def works_without_batch_dims():
        q2, k2, v2 = q[0, 0], k[0, 0], v[0, 0]  # (L, dk) 这种没有 batch 维的输入
        out, _ = sdpa(q2, k2, v2)
        assert_close(out, F.scaled_dot_product_attention(q2, k2, v2),
                     "无 batch 维输入的 out（只对最后两维操作，别写死维度数）")

    check("输出形状正确", shapes)
    check("注意力权重每行和为 1", rows_sum_to_one)
    check("与 PyTorch 官方结果一致（无 mask）", match_no_mask)
    check("与 PyTorch 官方结果一致（有 mask）", match_with_mask)
    check("支持任意前置维度", works_without_batch_dims)


# ----------------------------------------------------------------------------- Step 2
def step2():
    print("\n[Step 2] make_causal_mask")
    expected = torch.tensor([[1, 0, 0, 0],
                             [1, 1, 0, 0],
                             [1, 1, 1, 0],
                             [1, 1, 1, 1]], dtype=torch.bool)

    def values():
        m = A.make_causal_mask(4)
        assert isinstance(m, torch.Tensor), "应返回 Tensor"
        assert m.dtype == torch.bool, f"dtype 应为 torch.bool，得到 {m.dtype}"
        assert torch.equal(m, expected), f"mask 内容不对:\n{m.int()}\n期望:\n{expected.int()}"

    def causal_attention():
        torch.manual_seed(0)
        q, k, v = torch.randn(2, 6, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 8)
        out, w = unpack(A.scaled_dot_product_attention(q, k, v, A.make_causal_mask(6)),
                        "scaled_dot_product_attention")
        assert_close(w[:, 0, 0], torch.ones(2), "第 0 个位置只能看自己，权重应为 1")
        assert w.triu(1).abs().max().item() == 0, "上三角（未来位置）的权重应全为 0"
        assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), "因果 attention 的 out")

    check("mask 内容正确", values)
    check("配合 Step 1 实现因果注意力", causal_attention)


# ----------------------------------------------------------------------------- Step 3
def build_pair(d_model, n_heads):
    """建立你的 MHA 和官方 MHA，并把你的权重拷给官方的，让两者参数完全一样"""
    mha = A.MultiHeadAttention(d_model, n_heads)
    for name in ["W_q", "W_k", "W_v", "W_o"]:
        assert isinstance(getattr(mha, name, None), nn.Linear), f"缺少 self.{name}（应为 nn.Linear）"
    ref = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([mha.W_q.weight, mha.W_k.weight, mha.W_v.weight]))
        ref.in_proj_bias.copy_(torch.cat([mha.W_q.bias, mha.W_k.bias, mha.W_v.bias]))
        ref.out_proj.weight.copy_(mha.W_o.weight)
        ref.out_proj.bias.copy_(mha.W_o.bias)
    return mha.eval(), ref.eval()


def step3():
    print("\n[Step 3] MultiHeadAttention")
    D, H, B, L, S = 32, 4, 2, 5, 9

    def init_ok():
        torch.manual_seed(0)
        mha, ref = build_pair(D, H)
        n_mine = sum(p.numel() for p in mha.parameters())
        n_ref = sum(p.numel() for p in ref.parameters())
        assert n_mine == n_ref, f"参数量 {n_mine}，期望 {n_ref}（= 4*D*D + 4*D）"

    def split_merge():
        torch.manual_seed(0)
        mha, _ = build_pair(D, H)
        x = torch.randn(B, L, D)
        s = mha.split_heads(x)
        assert s.shape == (B, H, L, D // H), f"split_heads 输出形状 {tuple(s.shape)}，期望 {(B, H, L, D // H)}"
        dh = D // H
        assert torch.equal(s[1, 2, 3], x[1, 3, 2 * dh:3 * dh]), \
            "split_heads 切法不对: 第 h 个头应拿到 x[..., h*Dh:(h+1)*Dh]（检查 view 的形状和 transpose 的维度）"
        m = mha.merge_heads(s)
        assert m.shape == x.shape, f"merge_heads 输出形状 {tuple(m.shape)}，期望 {tuple(x.shape)}"
        assert torch.equal(m, x), "merge_heads(split_heads(x)) 应该等于 x"

    def self_attn():
        torch.manual_seed(0)
        mha, ref = build_pair(D, H)
        x = torch.randn(B, L, D)
        out, w = unpack(mha(x, x, x), "forward")
        r_out, r_w = ref(x, x, x, need_weights=True, average_attn_weights=False)
        assert_close(out, r_out, "self-attention 的 out")
        assert_close(w, r_w, "self-attention 每个头的 weights")

    def causal_self_attn():
        torch.manual_seed(0)
        mha, ref = build_pair(D, H)
        x = torch.randn(B, L, D)
        mask = A.make_causal_mask(L)
        out, w = unpack(mha(x, x, x, mask=mask), "forward")
        # 注意: nn.MultiheadAttention 的 bool mask 约定相反（True = 屏蔽），所以传 ~mask
        r_out, r_w = ref(x, x, x, attn_mask=~mask, need_weights=True, average_attn_weights=False)
        assert_close(out, r_out, "因果 self-attention 的 out")
        assert_close(w, r_w, "因果 self-attention 的 weights")

    def cross_attn():
        torch.manual_seed(0)
        mha, ref = build_pair(D, H)
        tgt, mem = torch.randn(B, L, D), torch.randn(B, S, D)  # L != S
        out, w = unpack(mha(tgt, mem, mem), "forward")
        r_out, r_w = ref(tgt, mem, mem, need_weights=True, average_attn_weights=False)
        assert_close(out, r_out, f"cross-attention 的 out（query 长 {L}，key 长 {S}）")
        assert_close(w, r_w, "cross-attention 的 weights")

    check("__init__ 参数量正确", init_ok)
    check("split_heads / merge_heads 正确", split_merge)
    check("self-attention 与官方一致", self_attn)
    check("带因果 mask 的 self-attention 与官方一致", causal_self_attn)
    check("cross-attention（L≠S）与官方一致", cross_attn)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    steps = {"1": step1, "2": step2, "3": step3}
    chosen = sys.argv[1:] or list(steps)
    for s in chosen:
        if s not in steps:
            sys.exit(f"未知的 step: {s}，可选 1 / 2 / 3")
        steps[s]()

    n_pass, n_all = sum(results), len(results)
    print(f"\n通过 {n_pass}/{n_all}")
    if n_pass == n_all and not sys.argv[1:]:
        print("🎉 全部通过！下一步: python visualize.py")
