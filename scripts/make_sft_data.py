"""從中文維基語料「自抽」指令微調(SFT)資料：把「X是Y。」轉成「問：什麼是X？／答：X是Y。」

好處：用字都在現有 vocab 內、不用另外下載，而且就是 self-instruct 的雛形。
維基條目開頭通常就是一句定義句，正好拿來當「指令→回答」配對。

輸出 JSONL：每行 {"q": ..., "a": ...}
  python scripts/make_sft_data.py --max 6000
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import clean as C   # noqa: E402

# X是Y。  X：1–14 字（標題）、Y：8–80 字（定義），到句號為止
_DEF = re.compile(r"^(.{1,14}?)是(.{8,80}?。)")
_BAD = re.compile(r"[，。、；：（）「」『』\s0-9A-Za-z]")   # 標題裡不該有這些


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/zhwiki.txt")
    ap.add_argument("--doc_sep", default="<|doc|>")
    ap.add_argument("--out", default="artifacts/sft.jsonl")
    ap.add_argument("--max", type=int, default=6000)
    args = ap.parse_args()

    docs = Path(args.input).read_text(encoding="utf-8").split(args.doc_sep)
    seen, pairs = set(), []
    for d in docs:
        t = C.normalize_text(d).strip()
        m = _DEF.match(t)
        if not m:
            continue
        title, defn = m.group(1), m.group(2)
        if _BAD.search(title) or title in seen:      # 標題乾淨、不重複
            continue
        seen.add(title)
        pairs.append({"q": f"什麼是{title}？", "a": f"{title}是{defn}"})
        if len(pairs) >= args.max:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"產出 {len(pairs)} 筆 SFT 配對 -> {out}")
    for p in pairs[:3]:
        print(f"  問：{p['q']}  答：{p['a'][:40]}…")


if __name__ == "__main__":
    main()
