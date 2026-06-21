"""從中文維基「自抽」DPO 偏好資料：把同一個問題配一組「好/壞」回答。

DPO 需要「偏好對」：同一個 prompt，一個 chosen（較好）、一個 rejected（較差）。
本檔提供兩種偏好「軸」，用來對照「偏好在模型容量內 vs 超出容量」會怎樣：

  --mode topic（預設，難）：打 SFT 缺陷「張冠李戴」
    chosen  ：X是<X 自己的定義>     ← 內容對題（on-topic）
    rejected：X是<別條目 Z 的定義>  ← 主詞對、內容是別人的（off-topic）
    要分辨需要「標題↔內容」的語義綁定 → 8M char 模型學不動、只會背訓練對。

  --mode format（易）：連貫 vs 退化重複
    chosen  ：X是<X 的定義>         ← 正常通順
    rejected：X是<某字反覆>          ← 卡住的重複迴圈（壞 LM 的典型產出）
    「避免重複」是低階統計特徵 → 模型學得動、會類推到沒見過的題。

對照組設計重點：兩種模式 chosen/rejected 都同主詞開頭、長度相近 →
唯一變因是「內容對題 / 是否退化」，不被長度等混淆。

輸出 JSONL：每行 {"prompt": ..., "chosen": ..., "rejected": ...}
  python scripts/make_dpo_data.py --mode topic
  python scripts/make_dpo_data.py --mode format --out artifacts/dpo_format.jsonl \
      --heldout artifacts/dpo_format_heldout.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import clean as C   # noqa: E402

# X是Y。  X：1–14 字（標題）、Y：8–60 字（定義本體），到句號為止
_DEF = re.compile(r"^(.{1,14}?)是(.{8,60}?。)")
_BAD = re.compile(r"[，。、；：（）「」『』\s0-9A-Za-z]")   # 標題裡不該有這些


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/zhwiki.txt")
    ap.add_argument("--doc_sep", default="<|doc|>")
    ap.add_argument("--out", default="artifacts/dpo.jsonl")
    ap.add_argument("--heldout", default="artifacts/dpo_heldout.jsonl")
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--max", type=int, default=6000)
    ap.add_argument("--mode", choices=["topic", "format"], default="topic",
                    help="topic=張冠李戴(難,學不動)；format=連貫vs退化重複(易,會類推)")
    ap.add_argument("--stride", type=int, default=337,
                    help="topic 模式：rejected 取往後第 stride 個條目的定義（質數避免撞週期）")
    args = ap.parse_args()

    docs = Path(args.input).read_text(encoding="utf-8").split(args.doc_sep)
    seen, items = set(), []          # items = (title, body)，body 含結尾句號
    for d in docs:
        t = C.normalize_text(d).strip()
        m = _DEF.match(t)
        if not m:
            continue
        title, body = m.group(1), m.group(2)
        if _BAD.search(title) or title in seen:
            continue
        seen.add(title)
        items.append((title, body))
        if len(items) >= args.max:
            break

    n = len(items)
    pairs = []
    for i, (title, body) in enumerate(items):
        chosen = f"{title}是{body}"
        if args.mode == "topic":
            # rejected 借「往後第 stride 個」條目的定義本體（張冠李戴）
            wrong_title, wrong_body = items[(i + args.stride) % n]
            if wrong_title == title or wrong_body == body:
                continue             # 萬一撞到就跳過，保證內容真的不同
            rejected = f"{title}是{wrong_body}"
        else:  # format：把本體換成「前 2 字反覆」湊到相近長度的退化迴圈
            seed = body[:2] if len(body) >= 2 else (body or "的")
            reps = max(1, (len(body) - 1) // len(seed))
            rejected = f"{title}是" + (seed * reps) + "。"
        pairs.append({
            "prompt": f"什麼是{title}？",
            "chosen": chosen,
            "rejected": rejected,
        })

    # 切 train / held-out（held-out 用來驗「偏好有沒有類推到沒訓過的題」）
    n_held = int(len(pairs) * args.heldout_frac)
    held, train = pairs[:n_held], pairs[n_held:]

    def dump(path, rows):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(args.out, train)
    dump(args.heldout, held)
    print(f"產出 {len(pairs)} 組偏好對：train {len(train)} -> {args.out}、"
          f"held-out {len(held)} -> {args.heldout}")
    for p in train[:2]:
        print(f"  問：{p['prompt']}")
        print(f"    ✅ chosen   : {p['chosen'][:42]}…")
        print(f"    ❌ rejected : {p['rejected'][:42]}…")


if __name__ == "__main__":
    main()
