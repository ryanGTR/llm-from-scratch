"""跑 mini BPE 並輸出「可審核的監控數據」：每步合併 log + CSV + JSON 報表。

用法：python scripts/train_bpe.py --merges 500 --input data/raw/input.txt
產物（artifacts/）：
  bpe_merges.csv   每一步合併的完整審核紀錄（step, 左, 右, 合併出, 頻率, vocab, 序列長）
  bpe_report.json  彙總：壓縮率、vocab 成長、學到的詞片段、樣本 tokenize
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bpe import train_bpe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/input.txt")
    ap.add_argument("--merges", type=int, default=500)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    print(f"語料：{len(text):,} 字元，起始 vocab（不同字元數）計算中…")
    res = train_bpe(text, args.merges)
    log = res["log"]

    # ── 監控視圖：前 30 步 + 每 50 步里程碑（其餘進 CSV）──────────────
    print(f"\n起始 vocab = {res['base_vocab']}（純字元）")
    print("=" * 70)
    print(f"{'步':>4} {'合併（左 + 右 → 新）':<28} {'頻率':>8} {'vocab':>6} {'序列長':>10}")
    print("-" * 70)
    for e in log:
        if e["step"] <= 30 or e["step"] % 50 == 0 or e["step"] == len(log):
            pair = f"{e['pair'][0]!r}+{e['pair'][1]!r} → {e['merged']!r}"
            print(f"{e['step']:>4} {pair:<28} {e['freq']:>8,} "
                  f"{e['vocab_size']:>6} {e['seq_len']:>10,}")

    # ── 彙總報表 ──────────────────────────────────────────────────
    art = Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    final_len = log[-1]["seq_len"] if log else len(text)
    final_vocab = log[-1]["vocab_size"] if log else res["base_vocab"]
    # 學到的「詞片段」：挑最長的幾個 token（通常是常見詞）
    learned = sorted((v for v in res["vocab"].values() if len(v) > 1),
                     key=len, reverse=True)[:25]

    sample = "ROMEO: But soft, what light"
    # 用 merges 把 sample 也 tokenize 一下（套用學到的合併）
    from src.bpe import merge as _merge
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    sids = [stoi.get(c, 0) for c in sample]
    for pair, nid in res["merges"].items():
        sids = _merge(sids, pair, nid)
    sample_pieces = [res["vocab"][i] for i in sids]

    report = {
        "orig_chars": len(text),
        "base_vocab": res["base_vocab"],
        "num_merges": len(log),
        "final_vocab": final_vocab,
        "final_seq_len": final_len,
        "compression": round(final_len / len(text), 3),   # 越小越省
        "chars_per_token": round(len(text) / final_len, 2),
        "learned_fragments_top25": learned,
        "sample_text": sample,
        "sample_tokens": sample_pieces,
        "sample_char_len": len(sample),
        "sample_bpe_len": len(sample_pieces),
    }
    (art / "bpe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))

    with open(art / "bpe_merges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "left", "right", "merged", "freq", "vocab_size", "seq_len"])
        for e in log:
            w.writerow([e["step"], e["pair"][0], e["pair"][1], e["merged"],
                        e["freq"], e["vocab_size"], e["seq_len"]])

    print("=" * 70)
    print(f"壓縮：{len(text):,} 字元 → {final_len:,} token "
          f"（每 token 約 {report['chars_per_token']} 字元，vocab {res['base_vocab']}→{final_vocab}）")
    print(f"學到的詞片段（最長 25 個）：{learned}")
    print(f"樣本 {sample!r}：char {len(sample)} 個 → BPE {len(sample_pieces)} 個 {sample_pieces}")
    print(f"\n審核檔：{art}/bpe_merges.csv（全部 {len(log)} 步）、{art}/bpe_report.json")


if __name__ == "__main__":
    main()
