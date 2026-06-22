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
"""

import math
import re
import time
import zlib
from collections import Counter, defaultdict

random_state = 12345


def lcg():
    """自帶的可重現偽隨機（不依賴 Math.random，跨機一致）。"""
    x = random_state
    while True:
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        yield x


rng = lcg()


def rand_below(k):
    return next(rng) % k


# ---- 1) 造語料：把莎士比亞切成「文件」，注入重複與髒資料 ----
text = open("input.txt", encoding="utf-8").read()
paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
docs = paras[:400]                                   # 400 篇乾淨文件

# 1) 先注入「藏起來的髒語法」：~20% 文件混入 wiki 繁簡轉換語法（只佔每篇一小段）
DIRTY = "-{zh-tw:範例;zh-cn:范例}-"
dirty_truth = set()
for idx in range(len(docs)):
    if rand_below(100) < 20:
        docs[idx] = docs[idx][:30] + DIRTY + docs[idx][30:]
        dirty_truth.add(idx)

# 2) 再注入 40 組近似重複：整篇複製、只改 2 個字（高相似但非完全相同）
dup_truth = set()
for _ in range(40):
    i = rand_below(400)                              # 從原始 400 篇裡挑
    d = list(docs[i])
    for _ in range(2):
        if d:
            d[rand_below(len(d))] = " "
    docs.append("".join(d))
    dup_truth.add((i, len(docs) - 1))

N = len(docs)
print(f"語料：{N} 篇文件（含注入的 {len(dup_truth)} 組近似重複、"
      f"{len(dirty_truth)} 篇含髒語法）\n")


# ---- 2) 聚合指標：先算「總分」----
def char_entropy(s):
    c = Counter(s)
    tot = len(s)
    return -sum((v / tot) * math.log2(v / tot) for v in c.values())

avg_len = sum(len(d) for d in docs) / N
all_text = "".join(docs)
ent = char_entropy(all_text)
print("=== 聚合指標（總分）===")
print(f"  平均文件長度：{avg_len:6.0f} 字元")
print(f"  全語料字元熵：{ent:6.2f} bits/char")
print(f"  → 看起來資料健康 ✅（長度正常、熵正常）\n")


# ---- 3) 一條偵測器：把「看資料的眼睛」寫成程式 ----
detector = re.compile(r"-\{zh-[a-z]{2}:")
hits = {i for i, d in enumerate(docs) if detector.search(d)}
actual_dirty = {i for i, d in enumerate(docs) if DIRTY in d}    # 真正含髒語法的（含被複製繼承的）
print("=== 一條偵測器（看樣本後寫成規則）===")
print(f"  命中 markup 髒語法：{len(hits)}/{N} = {len(hits)/N*100:.1f}% 文件")
print(f"  真正含髒語法：{len(actual_dirty)} 篇（偵測器全抓到？ {hits == actual_dirty}）")
print(f"  → 聚合指標看不到（只佔每篇一小段、被平均稀釋），偵測器才現形\n")


# ---- 4) 去重：MinHash + LSH vs naive 兩兩比 ----
def shingles(s, k=5):
    s = re.sub(r"\s+", " ", s)
    return {s[i:i + k] for i in range(max(1, len(s) - k + 1))}

def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0

sh = [shingles(d) for d in docs]

# naive：所有 C(N,2) 對都算一次 Jaccard
t0 = time.perf_counter()
naive_pairs = 0
naive_dupes = set()
for i in range(N):
    for j in range(i + 1, N):
        naive_pairs += 1
        if jaccard(sh[i], sh[j]) > 0.5:
            naive_dupes.add((i, j))
t_naive = time.perf_counter() - t0

# MinHash + LSH
NUM_HASH, BANDS = 64, 16
ROWS = NUM_HASH // BANDS
# 64 組 (a,b)，把 shingle 的 hash 再打散
coeffs = [(rand_below(1 << 30) | 1, rand_below(1 << 30)) for _ in range(NUM_HASH)]
PRIME = (1 << 61) - 1

def minhash(shingle_set):
    # 用 crc32 當基底雜湊：跨 process 可重現（Python 內建 hash() 對字串會加鹽，不可重現）
    base = [zlib.crc32(x.encode("utf-8")) for x in shingle_set]
    sig = []
    for a, b in coeffs:
        sig.append(min((a * h + b) % PRIME for h in base) if base else 0)
    return sig

t0 = time.perf_counter()
sigs = [minhash(s) for s in sh]
buckets = defaultdict(list)
for i, sig in enumerate(sigs):
    for band in range(BANDS):
        key = (band, tuple(sig[band * ROWS:(band + 1) * ROWS]))
        buckets[key].append(i)
candidates = set()
for members in buckets.values():
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            candidates.add((members[a], members[b]))
lsh_dupes = {(i, j) for (i, j) in candidates if jaccard(sh[i], sh[j]) > 0.5}
t_lsh = time.perf_counter() - t0

# 對「注入的真重複」算 recall（這些 Jaccard 很高，是該抓到的硬目標）
naive_inj = len(naive_dupes & dup_truth)
lsh_inj = len(lsh_dupes & dup_truth)

print("=== 去重：naive vs MinHash+LSH ===")
print(f"  naive：比較 {naive_pairs:>6} 對（C(N,2)，$O(n^2)$），抓到 {len(naive_dupes)} 組，"
      f"{t_naive*1000:6.1f} ms")
print(f"  LSH  ：只細比 {len(candidates):>6} 個候選對，抓到 {len(lsh_dupes)} 組，"
      f"{t_lsh*1000:6.1f} ms")
print(f"  注入的 {len(dup_truth)} 組真重複——naive 抓到 {naive_inj}、LSH 抓到 {lsh_inj}"
      f"（高相似的都抓到了）")
print(f"  LSH 漏掉的 {len(naive_dupes - lsh_dupes)} 組是 Jaccard 卡在門檻附近的邊界對——"
      f"LSH 是**近似**法，拿一點 recall 換速度（要更高 recall 就加 band）")
print(f"  關鍵：LSH 候選只佔 naive 的 {len(candidates)/naive_pairs*100:.1f}%——"
      f"N 越大這個比例越懸殊，這就是 $O(n^2)\\to$ 近 $O(n)$ 的來源")
