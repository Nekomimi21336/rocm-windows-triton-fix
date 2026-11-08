# rocm-windows-triton-fix

**A temporary workaround for running Triton / `torch.compile` on Windows + AMD Radeon (ROCm).**
Windows ＋ AMD Radeon（ROCm）で Triton / `torch.compile` を動かすための、**暫定的な回避策**。

> [!WARNING]
> **自己責任でご利用ください / Use at your own risk.**
> - これは根本的な修正ではなく、**一時的な回避策**です。正しさを証明したものではありません。
> - 1台のマシン（RX 7900 XTX / gfx1100）で、限られたケースを確認しただけです。他の GPU・バージョン・モデルでの動作は保証しません。
> - AMD・Triton・PyTorch の公式な修正や推奨ではなく、それらのプロジェクトとは無関係です。
> - 使う場合は、同梱のテストや自分のモデルで、コンパイル前後の結果が一致することを必ず確認してください。
>
> - This is **not a fix**, only a provisional workaround. Its correctness is **not proven**.
> - Checked on a single machine (RX 7900 XTX / gfx1100) for a limited set of cases. No guarantee for other GPUs, versions or models.
> - Not an official or recommended solution from AMD, Triton or PyTorch, and not affiliated with them.
> - If you use it, verify that compiled and eager results match for your own workload.

---

## 日本語

### 何ができるか
AMD 公式の Windows ネイティブ PyTorch（ROCm 7.2.1）に、コミュニティ版の Triton for Windows（AMD 対応版）を組み合わせて、
`torch.compile` を Windows 上の Radeon で動かす。

そのままだと、一部のカーネル（例: ヘッドごとの RMSNorm ＋ SiLU ゲートの逆伝播）のコンパイルで、
AMD 専用の最適化パス `TritonAMDFoldTrueCmpI` が `PassManager::run failed` で落ちる。
このリポジトリの `patch_triton.py` はそのパスを環境変数で飛ばせるようにする。

### 検証した環境
| 項目 | バージョン |
|---|---|
| OS | Windows 11 Pro |
| GPU | AMD Radeon RX 7900 XTX (gfx1100), 24GB |
| Python | 3.12（AMD の wheel は 3.12 のみ） |
| PyTorch | 2.9.1+rocm7.2.1（AMD 公式、repo.radeon.com） |
| Triton | triton-windows 3.5.1.post24（PyPI、PyTorch 2.9 対応版） |
| ビルド環境 | Visual Studio 2022 Build Tools / Windows SDK 10.0.26100 |

### セットアップ
1. **AMD 公式の手順で ROCm 版 PyTorch を入れる**（Python 3.12 の仮想環境に）
   手順と wheel の URL は AMD のドキュメント「Install PyTorch for Windows」（rocm.docs.amd.com の Radeon / Ryzen 向け）を参照。
   ROCm SDK（`rocm_sdk_core`, `rocm_sdk_devel`, `rocm_sdk_libraries_custom`, `rocm`）と `torch` の wheel を入れる。
2. **Triton for Windows を入れる**（PyTorch 2.9 には Triton 3.5 系）
   ```
   pip install "triton-windows==3.5.1.post24"
   ```
3. **ROCm SDK を展開する**
   ```
   rocm-sdk init
   ```
4. **パッチを当てる**（triton を入れ直したら再実行）
   ```
   python patch_triton.py
   ```
5. **自分のスクリプトの先頭で `rocm_env` を読み込む**（torch より先に）
   ```python
   import rocm_env   # ROCM_HOME, CC=clang-cl などを設定し、FoldTrueCmpI をスキップ
   import torch
   ```

### 動作確認
```
python tests/test_basic.py                    # Triton カーネルと torch.compile の基本動作
python tests/repro_fold_true_cmpi.py --no-fix # 不具合の再現（パッチ無効）→ PassManager::run failed
python tests/repro_fold_true_cmpi.py          # パッチ有効 → PASS
```
手元の結果:
```
--no-fix : note: Pipeline failed while executing [`TritonAMDFoldTrueCmpI` on 'builtin.module' operation]
           FAIL: InductorError RuntimeError: PassManager::run failed
fix on   : PASS: compiled fwd+bwd OK, grad rel err vs eager fp32 = 7.79e-03   (bf16 の通常の誤差範囲)
```
また、実際の言語モデル（Transformer 0.11B、線形注意系モデル 0.12B）でも、fp32 ではコンパイル前後の勾配が
相対誤差 1e-6〜1e-7 で一致することを確認した。

### 症状と回避策
- 落ちる箇所: `TritonAMDFoldTrueCmpI`（`triton/backends/amd/compiler.py` の `make_ttgir` 内 `add_fold_true_cmpi`）
- 発生したケース: torch.compile が生成した、転置を含むヘッドごとのリダクションと SiLU の勾配を融合した逆伝播カーネル
- 回避策: このパスを実行しないようにする。このパスは「常に真と分かっている整数比較を畳み込む」最適化と理解しており、
  飛ばしても計算の意味は変わらないと考えている。ただし**これはあくまで推測と手元の確認に基づくもので、保証ではない**
- 考えられる副作用: 最適化が1つ減るため、性能がわずかに落ちる可能性がある。確認していないケースで別の問題が起きる可能性も否定できない
- 参考資料: Triton が出力した MLIR の再現データを [`repro/fold_true_cmpi.mlir`](repro/fold_true_cmpi.mlir) に置いている（ユーザー名を含むパスは伏せてある）

### 回避策を外すには
- 一時的に無効化: 環境変数 `TRITON_AMD_SKIP_FOLD_TRUE_CMPI=0` を設定する（`rocm_env.py` より先に設定すること）
- 完全に元に戻す: `pip install --force-reinstall "triton-windows==3.5.1.post24"` で triton を入れ直す
- 新しいバージョンの triton-windows では、この問題が解消されている可能性がある。その場合はこの回避策は不要

### その他の注意
- **venv の `python.exe` はランチャー**で、本体の Python を子プロセスとして起動する。`timeout` などで親だけを止めると子が残る
- プロジェクトのパスに空白があると、ROCm の `offload-arch.exe` 呼び出しで警告が出る（動作には今のところ影響なし）
- 初回のコンパイルは数十秒〜数分かかる（キャッシュ後は速い）
- AMD の公式ドキュメント上、Windows 版 PyTorch は「学習は未サポート」とされているが、手元では学習も動作した

---

## English

### What this is
A setup guide plus a small, **provisional** patch that let `torch.compile` (Inductor → Triton) run on an AMD Radeon GPU on
**native Windows**, using AMD's official ROCm 7.2.1 Windows PyTorch wheels and the community `triton-windows` package
(which has an AMD HIP backend). Use at your own risk (see the warning above).

Without the patch, compiling some kernels (e.g. the backward of a per-head RMSNorm fused with a SiLU gate) crashes in the
AMD-specific pass `TritonAMDFoldTrueCmpI` with `RuntimeError: PassManager::run failed`.
`patch_triton.py` makes that pass skippable via `TRITON_AMD_SKIP_FOLD_TRUE_CMPI=1` (set by default by `rocm_env.py`).

### Tested environment
Windows 11 Pro, RX 7900 XTX (gfx1100), Python 3.12, torch 2.9.1+rocm7.2.1 (AMD, repo.radeon.com),
triton-windows 3.5.1.post24 (PyPI), VS 2022 Build Tools, Windows SDK 10.0.26100.

### Setup
1. Install AMD's ROCm PyTorch for Windows into a Python 3.12 venv, following AMD's "Install PyTorch for Windows" guide
   (Radeon/Ryzen docs on rocm.docs.amd.com).
2. `pip install "triton-windows==3.5.1.post24"` (Triton 3.5 pairs with PyTorch 2.9)
3. `rocm-sdk init`
4. `python patch_triton.py` (idempotent; re-run after reinstalling triton-windows)
5. `import rocm_env` at the top of your script, before `import torch`.

### Verify
```
python tests/test_basic.py
python tests/repro_fold_true_cmpi.py --no-fix   # reproduces the crash
python tests/repro_fold_true_cmpi.py            # passes with the workaround
```
On real models (a 0.11B Transformer and a 0.12B linear-attention-style model), gradients with and without compile
match to 1e-6–1e-7 relative error in fp32.

### Why skipping the pass is expected to be harmless (not guaranteed)
As far as I understand, `TritonAMDFoldTrueCmpI` only folds integer comparisons that are known to be always true, i.e. it is
an optimization, so skipping it should not change semantics. **This is an assumption backed only by local checks, not a
proof.** Possible side effects: a small performance cost, and unknown behaviour in cases that were not tested.
The MLIR reproducer emitted by Triton is kept in [`repro/fold_true_cmpi.mlir`](repro/fold_true_cmpi.mlir) for reference
(user paths redacted).

### Turning the workaround off
- Temporarily: set `TRITON_AMD_SKIP_FOLD_TRUE_CMPI=0` (before importing `rocm_env`).
- Completely: `pip install --force-reinstall "triton-windows==3.5.1.post24"`.
- Newer triton-windows releases may no longer have this problem, in which case the workaround is unnecessary.

### Notes
- A venv's `python.exe` on Windows is a launcher that spawns the real interpreter; killing only the launcher leaves the child running.
- Spaces in the project path trigger a harmless warning from ROCm's `offload-arch.exe`.
- This is a personal, provisional workaround tested on a single machine. Not affiliated with AMD, Triton or PyTorch.

## License
MIT (this repository's scripts only), provided "AS IS" without warranty of any kind — see [LICENSE](LICENSE).
Triton, PyTorch and ROCm are under their own licenses and are not redistributed here.
本リポジトリのスクリプトは MIT ライセンスで、無保証（AS IS）で提供します。Triton・PyTorch・ROCm は含まれていません。
