"""跑資料品質偵測器，輸出可審報表 artifacts/data_quality_report.json + 印表格。

對「清洗後」的文件跑偵測 → 看哪些問題「漏過清洗」還留著（例如維基語法）。
用法：python scripts/quality_report.py --input data/raw/zhwiki.txt --doc_sep "<|doc|>"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.sources import load_documents       # noqa: E402
from src.data import clean as C                     # noqa: E402
from src.data.quality_report import quality_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/zhwiki.txt")
    ap.add_argument("--doc_sep", default=None)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--label", default="run", help="這次的標籤（如 before/after），存進歷史供面板做前後對照")
    args = ap.parse_args()

    print(f"載入 {args.input} …")
    docs = load_documents(args.input, doc_sep=args.doc_sep)
    texts = [C.normalize_text(d.text) for d in docs]   # 對「清洗後」跑偵測
    print(f"對 {len(texts)} 篇（清洗後）跑品質偵測 …\n")

    rep = quality_report(texts)
    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    (art / "data_quality_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2))

    # 存進歷史（帶 label）供監控面板做 before/after 對照；同 label 覆蓋、保留最近 10 次
    hist_path = art / "quality_history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    hist = [h for h in hist if h["label"] != args.label]
    hist.append({"label": args.label, "all_pass": rep["all_pass"],
                 "pct": {r["name"]: r["pct"] for r in rep["detectors"]}})
    hist_path.write_text(json.dumps(hist[-10:], ensure_ascii=False, indent=2))

    icon = {True: "✅", False: "❌"}
    print(f"{'偵測項':16s} {'命中':>7} {'佔比':>7} {'門檻':>6}  判定")
    print("-" * 60)
    for r in rep["detectors"]:
        print(f"{r['name']:16s} {r['hits']:>7} {r['pct']:>6.2f}% {r['threshold_pct']:>5.1f}%  "
              f"{icon[r['pass']]} {r['note']}")
    print("-" * 60)
    print(f"整批判定：{icon[rep['all_pass']]} {'全部通過' if rep['all_pass'] else '有項目超標，需處理'}")
    # 印第一個 fail 的樣本，讓你看到「長什麼樣」
    for r in rep["detectors"]:
        if not r["pass"] and r["samples"]:
            print(f"\n例（{r['name']}）問題樣本：")
            for s in r["samples"]:
                print(f"  · {s}")
            break
    print(f"\n報表：{art}/data_quality_report.json")


if __name__ == "__main__":
    main()
