"""Dedup 階段：去掉重複文件。重複資料會讓模型「死背」，傷害很大。

兩種重複：
  1. 完全相同（exact）    —— 用內容雜湊比對，O(n)，便宜。
  2. 幾乎相同（near-dup） —— 改幾個字、加標題的轉貼。MinHash + LSH 估相似度。

near-dup 用 **字元 n-gram**（中文無空白也適用）+ **MinHash + LSH**：
  - MinHash：每篇壓成 num_perm 維簽章；兩簽章相等格數的比例 ≈ Jaccard 相似度。
  - LSH（Locality-Sensitive Hashing）：把簽章切成 bands，同 band 雜湊到同桶的才當
    「候選對」去驗證 → 只比可能相似的，從 O(n²) 壓到接近 O(n)，能處理上萬篇。

Java 類比：exact = HashSet<String> 去重；near = 「模糊版 equals()」+ LSH 索引只比候選。
"""

import hashlib
import zlib
from dataclasses import dataclass

import numpy as np


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


# ---------- 2) Near-dup via MinHash + LSH ----------

_P = 2147483647   # 2^31-1（Mersenne 質數）：讓 a*h 不溢位 int64


def _shingles(text: str, k: int = 5) -> set[str]:
    """字元 n-gram：用 k 個「字元」一組（中文無空白也適用；英文也通用）。

    先把所有空白收斂成單一空格，讓碎片穩定。
    """
    s = " ".join(text.split())
    if len(s) <= k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def _make_perms(num_perm: int):
    """確定性產生 num_perm 組 (a, b)（固定 seed → 可重現）。"""
    rng = np.random.default_rng(1234)
    a = rng.integers(1, _P, size=num_perm, dtype=np.int64)
    b = rng.integers(0, _P, size=num_perm, dtype=np.int64)
    return a, b


def _minhash(shingles: set[str], a, b):
    """MinHash 簽章（向量化）：每個 shingle 只 hash 一次(crc32)，再用 num_perm 組
    (a,b) 各取所有 shingle 的 min((a·h+b) mod P)。比「每 perm 各 hash 一次」快很多。
    """
    if not shingles:
        return np.zeros(len(a), dtype=np.int64)
    hs = np.fromiter((zlib.crc32(s.encode("utf-8")) % _P for s in shingles),
                     dtype=np.int64, count=len(shingles))
    return ((a[:, None] * hs[None, :] + b[:, None]) % _P).min(axis=1)


def _sig_similarity(a, b) -> float:
    return float(np.mean(a == b))


@dataclass
class NearDupConfig:
    threshold: float = 0.6   # 相似度 >= 門檻就算重複
    k: int = 5               # 字元 n-gram 大小
    num_perm: int = 64       # 簽章長度
    bands: int = 16          # LSH band 數（num_perm 要能被整除；r=num_perm/bands）


def near_dedup(texts: list[str], cfg: NearDupConfig) -> tuple[list[int], int]:
    """MinHash + LSH 去近似重複。LSH 只比「同桶的候選對」→ 接近 O(n)，可處理上萬篇。

    回傳 (保留 index, 丟掉幾筆)。每篇逐一進來：用 band 桶找出已保留的相似候選，
    只在候選裡用簽章相似度驗證；沒撞到就保留並把自己加進桶。
    """
    a, b = _make_perms(cfg.num_perm)
    r = cfg.num_perm // cfg.bands
    buckets: dict = {}          # (band, band_key bytes) -> [已保留 doc index]
    kept_sigs: dict = {}        # doc index -> 簽章
    keep: list[int] = []
    for i, t in enumerate(texts):
        sig = _minhash(_shingles(t, cfg.k), a, b)
        cand: set[int] = set()
        band_keys = []
        for bnd in range(cfg.bands):
            key = (bnd, sig[bnd * r:(bnd + 1) * r].tobytes())
            band_keys.append(key)
            cand.update(buckets.get(key, ()))
        if any(_sig_similarity(sig, kept_sigs[c]) >= cfg.threshold for c in cand):
            continue                      # 跟某個候選太像，丟掉
        keep.append(i)
        kept_sigs[i] = sig
        for key in band_keys:
            buckets.setdefault(key, []).append(i)
    return keep, len(texts) - len(keep)
