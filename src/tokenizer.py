"""Character-level tokenizer — 最簡單的 tokenizer，一個字元一個 id。

Java 類比：把 String 轉成 int[]（serialize），模型只認得數字。
真實 LLM 用的是 BPE（subword）tokenizer，但 char-level 最好懂，
而且自己手刻一次你就懂「token 到底是什麼」了。
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
        self.itos = {i: ch for i, ch in enumerate(chars)}   # int -> string

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # 掃過全文，蒐集出現過的字元當作 vocabulary
        chars = sorted(set(text))
        return cls(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.chars, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        chars = json.loads(Path(path).read_text())
        return cls(chars)


def load_tokenizer(path: str | Path):
    """自動辨識存檔是 char 還是 bpe，回傳對應的 tokenizer。

    下游階段（eval / generate / viz）統一用這個載入，就不必管當初用哪種。
    判斷依據：char 存的是 list、bpe 存的是 dict（含 type:"bpe"）。
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and data.get("type") == "bpe":
        from src.bpe import BPETokenizer
        return BPETokenizer.load(path)
    return CharTokenizer(data)
