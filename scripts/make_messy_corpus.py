"""產一份「故意很髒」的小語料，讓你看到清洗/去重真的有在動。

裡面塞了：正常文、完全重複、近似重複（改幾個字）、HTML 標籤、控制字元、
太短、太重複（aaaa...）、符號洗版。跑 prepare_data 後對照 data_report.json
就能看到每一關砍掉哪些。

用法：python scripts/make_messy_corpus.py   ->  data/raw/demo/*.txt
"""

from pathlib import Path

GOOD = [
    "The quick brown fox jumps over the lazy dog near the riverbank every morning.",
    "Linux 的虛擬檔案系統（VFS）讓不同檔案系統共用同一組 read/write 介面。",
    "A page fault happens when a process touches a page that is not currently mapped.",
    "資料品質決定模型上限：garbage in, garbage out 在語言模型上特別明顯。",
    "Transformers replaced recurrence with self-attention, enabling parallel training.",
]

DOCS: dict[str, str] = {}
for i, g in enumerate(GOOD):
    DOCS[f"good_{i}.txt"] = g

# 完全重複（跟 good_0 一模一樣，只多了空白）
DOCS["dup_exact.txt"] = "The quick brown fox jumps over the lazy   dog near the riverbank every morning."

# 一對「近似重複」的長文：B 只是把 A 加個轉貼標題、改兩三個字。
# near-dup（MinHash）就是要抓這種——exact dedup 抓不到，因為一字之差雜湊就全變。
_PARAGRAPH = (
    "Self-attention lets every token in a sequence look at every other token and "
    "decide how much to focus on each one. This is computed with queries, keys, and "
    "values: the dot product of a query and a key gives an attention score, the scores "
    "are normalized with a softmax, and the result is a weighted sum of the values. "
    "Because all positions are processed at the same time, training parallelizes well "
    "on a GPU, which is the main reason transformers scaled far better than recurrent "
    "networks for large language models."
)
DOCS["near_a.txt"] = _PARAGRAPH
DOCS["near_b.txt"] = (
    "Reposted from our blog. "
    + _PARAGRAPH.replace("focus on each one", "weight each one")
                .replace("scaled far better", "scaled much better")
)
# HTML 垃圾（清洗後應該還算正常文）
DOCS["html.txt"] = "<div class='x'><p>Self-attention</p> lets every <b>token</b> look at all others.</div>"
# 控制字元亂碼
DOCS["ctrl.txt"] = "Hello\x00\x07 world\x1f this line had control bytes inside it for sure okay."
# 太短（會被 quality 丟）
DOCS["short.txt"] = "hi"
# 太重複（會被 quality 丟）
DOCS["repeat.txt"] = "a" * 200
# 符號洗版（會被 quality 丟）
DOCS["symbols.txt"] = "###@@@$$$%%%^^^&&&***!!!###@@@$$$%%%^^^&&&***!!!###@@@$$$%%%"


def main():
    out = Path("data/raw/demo")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (out / name).write_text(text, encoding="utf-8")
    print(f"寫了 {len(DOCS)} 篇到 {out}/")


if __name__ == "__main__":
    main()
