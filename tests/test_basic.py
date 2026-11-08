"""Basic check: a hand-written Triton kernel and torch.compile both run and give correct results.
基本チェック: 自作 Triton カーネルと torch.compile が動き、結果が正しいこと。

    python tests/test_basic.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rocm_env  # noqa: E402,F401
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def axpy_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2 + tl.load(y_ptr + offs, mask=m), mask=m)


def main():
    print(f"torch {torch.__version__} | triton {triton.__version__} | {torch.cuda.get_device_name(0)}")
    ok = True
    t = time.time()
    x = torch.randn(1_000_003, device="cuda"); y = torch.randn_like(x); o = torch.empty_like(x)
    axpy_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), BLOCK=1024)
    err = (o - (2 * x + y)).abs().max().item()
    print(f"[1] Triton kernel     max err {err:.1e}  ({time.time() - t:.1f}s incl. compile)")
    ok &= err == 0

    def f(a, b):
        return torch.nn.functional.silu(a) * b + a.pow(2).mean(-1, keepdim=True).rsqrt()
    t = time.time()
    a = torch.randn(64, 768, device="cuda"); b = torch.randn_like(a)
    err = (torch.compile(f)(a, b) - f(a, b)).abs().max().item()
    print(f"[2] torch.compile     max err {err:.1e}  ({time.time() - t:.1f}s incl. compile)")
    ok &= err < 1e-5
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
