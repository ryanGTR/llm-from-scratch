"""Streaming 抓中文維基（zhwiki），只取到設定的 MB 上限，不下整包。

文章之間用分隔符 <|doc|> 串成「單一檔」，之後 prepare_data 用 --doc_sep 切回
每篇文章＝去重/品質過濾的「文件」單位（避免寫上萬個小檔）。

用法：python scripts/get_chinese_data.py --max_mb 100
"""

import argparse
from pathlib import Path

from datasets import load_dataset

SEP = "\n<|doc|>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_mb", type=float, default=100.0)
    ap.add_argument("--max_docs", type=int, default=200000)
    ap.add_argument("--out", default="data/raw/zhwiki.txt")
    args = ap.parse_args()

    print("連線 HuggingFace，streaming 中文維基（wikimedia/wikipedia 20231101.zh）…")
    ds = load_dataset("wikimedia/wikipedia", "20231101.zh",
                      split="train", streaming=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    budget = args.max_mb * 1024 * 1024
    total = 0
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex["text"]
            f.write(text)
            f.write(SEP)
            total += len(text.encode("utf-8"))
            n += 1
            if n % 2000 == 0:
                print(f"  {n} 篇，{total/1e6:.1f} MB …")
            if total >= budget or n >= args.max_docs:
                break

    print(f"\n完成：{n} 篇文章、{total/1e6:.1f} MB -> {out}")
    print(f"文件分隔符 = {SEP!r}（prepare_data 用 --doc_sep '<|doc|>' 切回每篇）")
    sample = out.read_text(encoding="utf-8")[:200].replace("\n", " ")
    print(f"開頭樣本：{sample}")


if __name__ == "__main__":
    main()
    # datasets 的背景 streaming thread 在直譯器關閉時會丟 PyGILState 雜訊；
    # 檔案此時已寫完關好，直接 os._exit 乾淨離開、避開那個無害的崩潰訊息。
    import os
    os._exit(0)
