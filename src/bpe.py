"""Mini BPE trainer — 看它「合併最常見的相鄰配對」一步步學出子詞。

這是 C 的階段 1：純演算法，不接 pipeline。核心三個函式：
  get_pair_counts  數所有相鄰配對出現幾次
  merge            把某個配對全部替換成一個新 token id
  train_bpe        重複「找最常見配對 → 合併」，並記下每一步（給你審核）

Java 類比：就是一個貪婪壓縮器——每輪把「最常一起出現的兩塊」黏成一塊，
像在資料裡長出常見詞片段。純 stdlib。
"""

from collections import Counter


def get_pair_counts(ids: list[int]) -> Counter:
    """數每個「相鄰配對」出現幾次。"""
    counts: Counter = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把序列裡所有的 pair 替換成 new_id。"""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, num_merges: int) -> dict:
    """訓練 BPE：從字元開始，做 num_merges 次「合併最常見配對」。

    回傳 dict，含 vocab（id->字串）、merges（配對->新id）、最終 ids、
    以及 log（每一步的審核紀錄）。
    """
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab: dict[int, str] = {i: c for i, c in enumerate(chars)}
    ids = [stoi[c] for c in text]

    merges: dict[tuple[int, int], int] = {}
    log: list[dict] = []
    base = len(chars)
    next_id = base

    for k in range(num_merges):
        counts = get_pair_counts(ids)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:               # 沒有重複配對可合併了，提早收手
            break
        ids = merge(ids, pair, next_id)
        merges[pair] = next_id
        vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
        log.append({
            "step": k + 1,
            "pair": [vocab[pair[0]], vocab[pair[1]]],
            "merged": vocab[next_id],
            "freq": freq,            # 這個配對合併前出現幾次
            "vocab_size": next_id + 1,
            "seq_len": len(ids),     # 合併後整段剩幾個 token
        })
        next_id += 1

    return {
        "vocab": vocab,
        "merges": merges,
        "ids": ids,
        "log": log,
        "base_vocab": base,
        "orig_len": len(text),
    }


class BPETokenizer:
    """跟 src/tokenizer.py 的 CharTokenizer 同介面的 BPE 版，可直接換進 pipeline。

    介面對齊：from_text / encode / decode / vocab_size / save / load。
    內部多存了 merges（合併規則，有順序）；encode 就是「照學到的順序套合併」。
    """

    def __init__(self, base_chars: list[str], merges: dict[tuple[int, int], int]):
        self.base_chars = base_chars
        self.stoi = {c: i for i, c in enumerate(base_chars)}
        self.merges = merges                       # (a,b) -> new_id，dict 保留順序
        # 重建 id -> 字串 的 vocab（base 字元 + 每條 merge 串起來）
        self.vocab: dict[int, str] = {i: c for i, c in enumerate(base_chars)}
        for (a, b), nid in merges.items():
            self.vocab[nid] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_text(cls, text: str, num_merges: int = 500) -> "BPETokenizer":
        res = train_bpe(text, num_merges)
        return cls(sorted(set(text)), res["merges"])

    def encode(self, s: str) -> list[int]:
        # 先轉成 base 字元 id（訓練時沒見過的字元直接跳過——簡化處理）
        ids = [self.stoi[c] for c in s if c in self.stoi]
        # 照「學到的順序」套用每條合併規則
        for pair, nid in self.merges.items():
            ids = merge(ids, pair, nid)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def save(self, path) -> None:
        import json
        from pathlib import Path
        # 只存 base_chars + merges（vocab 可重建），檔案小
        data = {
            "type": "bpe",
            "base_chars": self.base_chars,
            "merges": [[a, b, nid] for (a, b), nid in self.merges.items()],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        merges = {(a, b): nid for a, b, nid in data["merges"]}
        return cls(data["base_chars"], merges)
