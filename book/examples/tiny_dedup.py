"""tiny_dedup.py — 在 CPU 上親手看兩件真實資料的事：MinHash 去重、聚合指標的盲點。

書本 Ch4 的「💻 在你的機器上」配套程式。純標準庫、不需 torch、不需 GPU、幾秒跑完。
拿莎士比亞切成「文件」，動手注入近似重複與一種藏起來的髒資料，然後示範兩課：

  1) **MinHash + LSH 把去重從 $O(n^2)$ 降到接近 $O(n)$**：
     對拍——naive 兩兩比 vs LSH 只比候選對，找到同一批重複，但比較次數差一個數量級。
  2) **聚合指標的盲點**：平均長度、字元熵都說「資料健康 ✅」，
     但一條偵測器才抓得到 ~20% 文件殘留一種 markup 髒語法——
     **總分漂亮常常只是因為問題被平均掉了。**

用法：
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python tiny_dedup.py

CI / 出圖可 import 本檔的函式（build_corpus / dedup / detector_hits…），不會跑到 main。
"""

import math
import os
import re
import time
import zlib
from collections import Counter, defaultdict

DIRTY = "-{zh-tw:範例;zh-cn:范例}-"
DETECTOR = re.compile(r"-\{zh-[a-z]{2}:")
PRIME = (1 << 61) - 1


def make_rand(seed=12345):
    """自帶可重現偽隨機（不依賴 Math.random，跨機/跨 process 一致）。回傳 rand_below(k)。"""
    state = [seed]

    def rand_below(k):
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] % k
    return rand_below


def build_corpus(text, rand_below, n_docs=400, n_dupes=40):
    """切成文件 → 注入髒語法(~20%) → 注入 n_dupes 組近似重複。回傳 (docs, dup_truth)。"""
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
    docs = paras[:n_docs]
    for idx in range(len(docs)):                      # 1) 藏起來的髒語法
        if rand_below(100) < 20:
            docs[idx] = docs[idx][:30] + DIRTY + docs[idx][30:]
    dup_truth = set()                                 # 2) 近似重複：整篇複製、只改 2 字
    for _ in range(n_dupes):
        i = rand_below(n_docs)
        d = list(docs[i])
        for _ in range(2):
            if d:
                d[rand_below(len(d))] = " "
        docs.append("".join(d))
        dup_truth.add((i, len(docs) - 1))
    return docs, dup_truth


def char_entropy(s):
    c, tot = Counter(s), len(s)
    return -sum((v / tot) * math.log2(v / tot) for v in c.values())


def detector_hits(docs):
    return {i for i, d in enumerate(docs) if DETECTOR.search(d)}


def shingles(s, k=5):
    s = re.sub(r"\s+", " ", s)
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def minhash(shingle_set, coeffs):
    # 用 crc32 當基底雜湊：跨 process 可重現（Python 內建 hash() 對字串會加鹽，不可重現）
    base = [zlib.crc32(x.encode("utf-8")) for x in shingle_set]
    return [min((a * h + b) % PRIME for h in base) if base else 0 for a, b in coeffs]


def dedup(docs, rand_below, num_hash=64, bands=16, threshold=0.5):
    """回傳一個 dict：naive 與 LSH 各自的重複對、候選數、比較數、耗時。"""
    sh = [shingles(d) for d in docs]
    n = len(docs)
    rows = num_hash // bands

    t0 = time.perf_counter()
    naive_pairs, naive_dupes = 0, set()
    for i in range(n):
        for j in range(i + 1, n):
            naive_pairs += 1
            if jaccard(sh[i], sh[j]) > threshold:
                naive_dupes.add((i, j))
    t_naive = time.perf_counter() - t0

    coeffs = [(rand_below(1 << 30) | 1, rand_below(1 << 30)) for _ in range(num_hash)]
    t0 = time.perf_counter()
    sigs = [minhash(s, coeffs) for s in sh]
    buckets = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band in range(bands):
            buckets[(band, tuple(sig[band * rows:(band + 1) * rows]))].append(i)
    candidates = set()
    for members in buckets.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                candidates.add((members[a], members[b]))
    lsh_dupes = {(i, j) for (i, j) in candidates if jaccard(sh[i], sh[j]) > threshold}
    t_lsh = time.perf_counter() - t0

    return dict(naive_pairs=naive_pairs, naive_dupes=naive_dupes, candidates=candidates,
                lsh_dupes=lsh_dupes, t_naive=t_naive, t_lsh=t_lsh)


def main():
    n_docs = int(os.environ.get("TINY_DEDUP_DOCS", 400))   # CI 可調小
    text = open("input.txt", encoding="utf-8").read()
    rand = make_rand()
    docs, dup_truth = build_corpus(text, rand, n_docs=n_docs)
    N = len(docs)
    print(f"語料：{N} 篇文件（含注入的 {len(dup_truth)} 組近似重複、"
          f"{len(detector_hits(docs))} 篇含髒語法）\n")

    print("=== 聚合指標（總分）===")
    print(f"  平均文件長度：{sum(len(d) for d in docs) / N:6.0f} 字元")
    print(f"  全語料字元熵：{char_entropy(''.join(docs)):6.2f} bits/char")
    print("  → 看起來資料健康 ✅（長度正常、熵正常）\n")

    hits = detector_hits(docs)
    actual = {i for i, d in enumerate(docs) if DIRTY in d}
    print("=== 一條偵測器（看樣本後寫成規則）===")
    print(f"  命中 markup 髒語法：{len(hits)}/{N} = {len(hits) / N * 100:.1f}% 文件")
    print(f"  真正含髒語法：{len(actual)} 篇（偵測器全抓到？ {hits == actual}）")
    print("  → 聚合指標看不到（只佔每篇一小段、被平均稀釋），偵測器才現形\n")

    r = dedup(docs, rand)
    nd, ld, cand = r["naive_dupes"], r["lsh_dupes"], r["candidates"]
    print("=== 去重：naive vs MinHash+LSH ===")
    print(f"  naive：比較 {r['naive_pairs']:>6} 對（C(N,2)，O(n^2)），抓到 {len(nd)} 組，"
          f"{r['t_naive']*1000:6.1f} ms")
    print(f"  LSH  ：只細比 {len(cand):>6} 個候選對，抓到 {len(ld)} 組，"
          f"{r['t_lsh']*1000:6.1f} ms")
    print(f"  注入的 {len(dup_truth)} 組真重複——naive 抓到 {len(nd & dup_truth)}、"
          f"LSH 抓到 {len(ld & dup_truth)}（高相似的都抓到了）")
    print(f"  LSH 漏掉的 {len(nd - ld)} 組是 Jaccard 卡在門檻附近的邊界對——"
          f"LSH 是近似法，拿一點 recall 換速度（要更高 recall 就加 band）")
    print(f"  關鍵：LSH 候選只佔 naive 的 {len(cand) / r['naive_pairs'] * 100:.1f}%——"
          f"N 越大這個比例越懸殊，這就是 O(n^2)→ 近 O(n) 的來源")


if __name__ == "__main__":
    main()
