"""Collect 階段：把原始資料讀成一串「文件（document）」。

為什麼以「文件」為單位？因為清洗的品質過濾、去重都是對「一篇」做判斷
（這篇太短？這兩篇重複？）。把全部文字當一坨字串就沒辦法去重了。

Java 類比：像 Spring Batch 的 ItemReader——把來源變成一筆筆 record。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    text: str
    source: str                       # 來自哪個檔案，方便 debug / 追溯
    meta: dict = field(default_factory=dict)


def load_documents(path: str | Path, doc_sep: str | None = None) -> list[Document]:
    """讀入文件。

    - path 是資料夾：遞迴抓所有 *.txt，每個檔 = 一篇文件。
    - path 是檔案：
        doc_sep=None  -> 整個檔當成一篇（適合 shakespeare 這種連續長文）。
        doc_sep="..." -> 用分隔字串切成多篇（適合一個檔塞很多短文）。
    """
    p = Path(path)
    docs: list[Document] = []

    if p.is_dir():
        for f in sorted(p.rglob("*.txt")):
            text = f.read_text(encoding="utf-8", errors="replace")
            docs.append(Document(text=text, source=str(f)))
    elif p.is_file():
        text = p.read_text(encoding="utf-8", errors="replace")
        if doc_sep:
            for i, chunk in enumerate(text.split(doc_sep)):
                docs.append(Document(text=chunk, source=f"{p}#{i}"))
        else:
            docs.append(Document(text=text, source=str(p)))
    else:
        raise FileNotFoundError(f"找不到輸入：{p}")

    return docs
