"""Dedup 階段：去掉重複文件。重複資料會讓模型「死背」，傷害很大。

兩種重複：
  1. 完全相同（exact）       —— 用內容雜湊比對，O(n)，便宜。
  2. 幾乎相同（near-dup）    —— 改幾個字、加個標題的轉貼。用 MinHash 估相似度。

這裡的 near-dup 用「貪婪兩兩比對」O(n^2)，小語料夠用、好懂。
真實大規模 pipeline 會加 LSH（Locality-Sensitive Hashing）把它壓到接近 O(n)
——這正是之後拿 Rust 重寫、練系統程式的好題目。

Java 類比：exact dedup = 用 HashSet<String> 去重；
near-dup = 自己刻一個「模糊版 equals()」+ 相似度門檻。
"""

import hashlib
from dataclasses import dataclass


# ---------- 1) Exact dedup ----------

def _content_hash(text: str) -> str:
    # 正規化空白後再 hash，避免「只差幾個空白」被當成不同文件
    norm = " ".join(text.split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def exact_dedup(texts: list[str]) -> tuple[list[int], int]:
    """回傳 (要保留的 index 清單, 丟掉幾筆)。第一次出現的保留。"""
    seen: set[str] = set()
    keep: list[int] = []
    for i, t in enumerate(texts):
        h = _content_hash(t)
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep, len(texts) - len(keep)


# ---------- 2) Near-dup via MinHash ----------

def _shingles(text: str, k: int = 4) -> set[str]:
    """把文字切成 k 個詞一組的「碎片集合」。兩篇共享越多碎片就越像。"""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _minhash(shingles: set[str], num_perm: int = 64) -> list[int]:
    """MinHash 簽章：用 num_perm 個雜湊函數，各取所有 shingle 的最小雜湊值。

    神奇之處：兩個集合的簽章在每一格相等的機率 == 它們的 Jaccard 相似度。
    所以比「兩個 64 維簽章」就能估相似度，不用兩兩比對原始集合。
    """
    if not shingles:
        return [0] * num_perm
    sig = []
    for p in range(num_perm):
        salt = p.to_bytes(2, "big")
        best = min(
            int.from_bytes(hashlib.blake2b(s.encode("utf-8"), salt=salt, digest_size=8).digest(), "big")
            for s in shingles
        )
        sig.append(best)
    return sig


def _sig_similarity(a: list[int], b: list[int]) -> float:
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


@dataclass
class NearDupConfig:
    threshold: float = 0.6   # 相似度 >= 門檻就算重複（短文偏高易漏，長文較準）
    k: int = 4               # shingle 詞數
    num_perm: int = 64       # 簽章長度（越長越準、越慢）


def near_dedup(texts: list[str], cfg: NearDupConfig) -> tuple[list[int], int]:
    """貪婪去近似重複：逐篇比對已保留者的簽章，太像就丟。

    回傳 (保留 index, 丟掉幾筆)。注意：O(n^2 * num_perm)，大語料要改 LSH。
    """
    kept_sigs: list[list[int]] = []
    keep: list[int] = []
    for i, t in enumerate(texts):
        sig = _minhash(_shingles(t, cfg.k), cfg.num_perm)
        if any(_sig_similarity(sig, ks) >= cfg.threshold for ks in kept_sigs):
            continue                      # 跟某篇太像，丟掉
        kept_sigs.append(sig)
        keep.append(i)
    return keep, len(texts) - len(keep)
