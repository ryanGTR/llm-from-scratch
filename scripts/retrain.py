"""重訓迴圈（MLOps 外圈）：一鍵把整條鏈串起來自動跑。

  資料 pipeline → 訓練 → 評估(test) → 資料品質報表 → 註冊 → promotion gate → (可選)promote

觸發來源在真實世界是：drift 監控說「該重訓了」、排程、或新資料到。這支把「人工一步步跑」
變成「一個自動外圈」，而且尾端有 gate 把關——更爛的模型不會自動上線（回歸檢查）。

  python scripts/retrain.py --skip-data --max_iters 800            # 快速 demo（重用現有資料）
  python scripts/retrain.py --auto-promote                         # 過 gate 就自動上線
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import registry as R   # noqa: E402


def run(desc: str, cmd: list[str]):
    print(f"\n=== {desc} ===")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/zhwiki.txt")
    ap.add_argument("--doc_sep", default="<|doc|>")
    ap.add_argument("--max_iters", type=int, default=4000)
    ap.add_argument("--skip-data", action="store_true",
                    help="重用現有 artifacts 的資料，不重跑 prepare（demo 用）")
    ap.add_argument("--auto-promote", action="store_true",
                    help="過 gate 就自動 promote 上線")
    args = ap.parse_args()
    py = sys.executable

    if not args.skip_data:
        run("1/5 資料 pipeline",
            [py, "pipeline/01_prepare_data.py", "--input", args.input, "--doc_sep", args.doc_sep])
        run("品質報表（gate 要用）",
            [py, "scripts/quality_report.py", "--doc_sep", args.doc_sep, "--label", "retrain"])
    run("2/5 訓練",
        [py, "pipeline/02_train.py", "--max_iters", str(args.max_iters),
         "--block_size", "256", "--n_layer", "6", "--n_embd", "256", "--run_name", "retrain",
         "--use_rmsnorm", "--use_swiglu", "--use_rope", "--n_kv_head", "2", "--use_flash"])
    run("3/5 評估(test)", [py, "pipeline/03_eval.py", "--split", "test"])

    print("\n=== 4/5 註冊進 registry ===")
    entry = R.register()
    print(f"已註冊 {entry['short']}  test_loss={entry['metrics'].get('test_loss')}")

    print("\n=== 5/5 promotion gate ===")
    reg = R.load_registry()
    prod = next((e for e in reg if e.get("status") == "production"
                 and e["model_digest"] != entry["model_digest"]), None)
    reasons = R.gate_reasons(entry, prod)
    if reasons:
        print("❌ gate 擋下，不上線：" + "、".join(reasons))
        if prod:
            print(f"   （現行 production 維持 {prod['short']} 不變）")
    elif args.auto_promote:
        ok, msg = R.promote(entry["short"])
        print(("✅ 自動上線：" if ok else "❌ ") + msg)
    else:
        print(f"✅ 通過 gate，可上線。手動執行：python scripts/registry_cli.py promote {entry['short']}")


if __name__ == "__main__":
    main()
