"""Environment setup for Triton / torch.compile on Windows + AMD ROCm (AMD's native Windows PyTorch wheels).
Import this BEFORE importing torch or triton:   import rocm_env
Windows + ROCm で Triton / torch.compile を使うための環境変数設定。torch / triton より先に import すること。
"""
import os
import sys

_venv = os.path.dirname(os.path.dirname(sys.executable))
_sp = os.path.join(_venv, "Lib", "site-packages")
_devel = os.path.join(_sp, "_rocm_sdk_devel")   # created by `rocm-sdk init`
_core = os.path.join(_sp, "_rocm_sdk_core")
ROCM_HOME = _devel if os.path.isdir(_devel) else _core

os.environ.setdefault("ROCM_HOME", ROCM_HOME)
os.environ.setdefault("HIP_PATH", ROCM_HOME)
os.environ.setdefault("CC", "clang-cl")
os.environ.setdefault("CXX", "clang-cl")
os.environ.setdefault("DISTUTILS_USE_SDK", "1")
# Switch added by patch_triton.py: skip the crashing TritonAMDFoldTrueCmpI pass
# patch_triton.py で追加したスイッチ: クラッシュする FoldTrueCmpI パスをスキップする
os.environ.setdefault("TRITON_AMD_SKIP_FOLD_TRUE_CMPI", "1")
_bins = [os.path.join(ROCM_HOME, "lib", "llvm", "bin"), os.path.join(ROCM_HOME, "bin"),
         os.path.join(_core, "lib", "llvm", "bin"), os.path.join(_core, "bin")]
os.environ["PATH"] = os.pathsep.join([b for b in _bins if os.path.isdir(b)] + [os.environ.get("PATH", "")])
