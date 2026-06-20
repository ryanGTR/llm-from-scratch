"""Stage 1 — Data prep：原始資料 -> 乾淨、去重、tokenized 的訓練檔。

這是整條 LLM pipeline 裡「資料」這一棒，子流程：
  collect -> clean(normalize) -> quality filter -> exact dedup
         -> near dedup -> concat -> tokenize -> train/val split -> pack(.bin)

每一步都印「處理前/後」統計，並把整份報表寫到 artifacts/data_report.json，
讓你「看得到」每一關砍掉了什麼。純 Python、零依賴。

產物（artifacts/）：
  tokenizer.json · train.bin · val.bin · meta.json · data_report.json
  clean_corpus.txt（去重後合併的乾淨全文，方便你肉眼檢查）

用法：
  python pipeline/01_prepare_data.py --input data/raw/input.txt
  python pipeline/01_prepare_data.py --input data/raw/demo   # 一資料夾多篇
"""

import argparse
import json
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer import CharTokenizer            # noqa: E402
from src.data.sources import load_documents        # noqa: E402
from src.data import clean as C                     # noqa: E402
from src.data import dedup as D                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/input.txt",
                    help="檔案或資料夾")
    ap.add_argument("--doc_sep", default=None,
                    help="單檔內的文件分隔字串；不給則整檔當一篇")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--no_dedup", action="store_true", help="關閉去重（對照用）")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                    help="char=字元級；bpe=子詞級")
    ap.add_argument("--merges", type=int, default=500,
                    help="bpe 才用：合併幾次（vocab ≈ 字元數 + merges）")
    args = ap.parse_args()

    report: dict = {"stages": []}

    def step(name, before, after, **extra):
        dropped = before - after
        line = {"stage": name, "in": before, "out": after, "dropped": dropped, **extra}
        report["stages"].append(line)
        print(f"  {name:16s} {before:>6} -> {after:>6}  (丟 {dropped})"
              + (f"  {extra}" if extra else ""))

    # 1) Collect ------------------------------------------------------------
    print("[1] collect")
    docs = load_documents(args.input, doc_sep=args.doc_sep)
    n0 = len(docs)
    print(f"  讀入 {n0} 篇文件，共 {sum(len(d.text) for d in docs):,} 字元")

    # 2) Clean（normalize，逐篇修） -----------------------------------------
    print("[2] clean / normalize")
    for d in docs:
        d.text = C.normalize_text(d.text)

    # 3) Quality filter（丟爛文件） -----------------------------------------
    print("[3] quality filter")
    qcfg = C.QualityConfig()
    reasons: dict[str, int] = {}
    kept_docs = []
    for d in docs:
        ok, why = C.quality_check(d.text, qcfg)
        if ok:
            kept_docs.append(d)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    step("quality", len(docs), len(kept_docs), reasons=reasons)
    docs = kept_docs

    # 4) Exact dedup --------------------------------------------------------
    if not args.no_dedup:
        print("[4] exact dedup")
        keep, _ = D.exact_dedup([d.text for d in docs])
        before = len(docs)
        docs = [docs[i] for i in keep]
        step("exact_dedup", before, len(docs))

        # 5) Near dedup -----------------------------------------------------
        print("[5] near dedup (MinHash)")
        ncfg = D.NearDupConfig()
        keep, _ = D.near_dedup([d.text for d in docs], ncfg)
        before = len(docs)
        docs = [docs[i] for i in keep]
        step("near_dedup", before, len(docs), threshold=ncfg.threshold)

    # 6) Concat -> tokenize -------------------------------------------------
    print(f"[6] tokenize（{args.tokenizer}）")
    full = "\n\n".join(d.text for d in docs)
    if args.tokenizer == "bpe":
        # 直接用 train_bpe 的輸出 ids（避免再 encode 整段大語料，省時間）
        from src.bpe import train_bpe, BPETokenizer
        res = train_bpe(full, args.merges)
        ids = res["ids"]
        tok = BPETokenizer(sorted(set(full)), res["merges"])
    else:
        tok = CharTokenizer.from_text(full)
        ids = tok.encode(full)
    print(f"  {len(full):,} 字元 -> {len(ids):,} token，vocab_size = {tok.vocab_size}")

    # 7) Split + pack（純 Python 寫 uint16，x86 little-endian 與 numpy 相容） -
    print("[7] split + pack")
    n_val = max(1, int(len(ids) * args.val_frac))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "tokenizer.json")
    array("H", train_ids).tofile(open(out / "train.bin", "wb"))
    array("H", val_ids).tofile(open(out / "val.bin", "wb"))
    (out / "clean_corpus.txt").write_text(full, encoding="utf-8")

    meta = {
        "vocab_size": tok.vocab_size,
        "tokenizer": args.tokenizer,
        "docs_in": n0,
        "docs_out": len(docs),
        "total_chars": len(full),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "chars_per_token": round(len(full) / len(ids), 3),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    report["meta"] = meta
    (out / "data_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\n完成：{n0} 篇 -> {len(docs)} 篇，train={len(train_ids):,} "
          f"val={len(val_ids):,} token  -> {out}/")
    print(f"報表：{out}/data_report.json   乾淨全文：{out}/clean_corpus.txt")


if __name__ == "__main__":
    main()
