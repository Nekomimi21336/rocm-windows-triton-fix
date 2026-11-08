"""Minimal reproducer: `TritonAMDFoldTrueCmpI` crashes ("PassManager::run failed") while compiling the
backward pass of a per-head RMSNorm fused with a SiLU gate, on Windows + AMD (gfx1100) with triton-windows 3.5.1.

    python tests/repro_fold_true_cmpi.py            # with the workaround (pass skipped)  -> expected: PASS
    python tests/repro_fold_true_cmpi.py --no-fix   # without the workaround              -> expected: crash

最小再現コード: ヘッドごとの RMSNorm と SiLU ゲートを融合したカーネルの逆伝播で、
AMD 専用パス TritonAMDFoldTrueCmpI が "PassManager::run failed" で落ちる。
"""
import os
import sys
import tempfile

NO_FIX = "--no-fix" in sys.argv
os.environ["TRITON_AMD_SKIP_FOLD_TRUE_CMPI"] = "0" if NO_FIX else "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor_repro_")  # always compile from scratch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rocm_env  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class GatedHeadNorm(nn.Module):
    def __init__(self, d=768, n_head=12):
        super().__init__()
        self.H, self.dh = n_head, d // n_head
        self.q, self.k, self.v, self.g, self.o = (nn.Linear(d, d, bias=False) for _ in range(5))
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t(x).view(B, T, self.H, self.dh).transpose(1, 2) for t in (self.q, self.k, self.v))
        mask = torch.ones(T, T, device=x.device).tril()
        y = ((q @ k.transpose(-2, -1)) * mask) @ v                          # [B,H,T,dh] (linear-attention-like)
        y = y.transpose(1, 2).reshape(B, T, C)
        yh = y.view(B, T, self.H, self.dh)
        yh = yh * torch.rsqrt(yh.pow(2).mean(-1, keepdim=True) + 1e-6)      # per-head RMSNorm
        return self.o(yh.reshape(B, T, C) * self.w * F.silu(self.g(x)))      # SiLU gate


def main():
    torch.manual_seed(0)
    m = GatedHeadNorm().cuda()
    x = torch.randn(4, 512, 768, device="cuda", requires_grad=True)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | fix={'off' if NO_FIX else 'on'}", flush=True)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = torch.compile(m)(x)
        out.float().sum().backward()
        torch.cuda.synchronize()
    except Exception as e:
        print("FAIL:", type(e).__name__, str(e).splitlines()[0][:200])
        sys.exit(1)
    # compare with eager (fp32) to make sure the compiled result is also correct
    x2 = x.detach().clone().requires_grad_()
    ref = m(x2)
    ref.sum().backward()
    err = ((x.grad - x2.grad).norm() / x2.grad.norm()).item()
    print(f"PASS: compiled fwd+bwd OK, grad rel err vs eager fp32 = {err:.2e}")


if __name__ == "__main__":
    main()
