"""tiny_drift.py — 在 CPU 上親手量「模型會腐壞」：PSI 抓資料漂移、shadow 比一致率。

書本 Ch8（會腐壞的系統）的「💻 在你的機器上」配套程式。純標準庫、不需 torch、瞬間跑完。
模型上線後，世界會變——線上請求的分布會慢慢偏離訓練分布（data drift）。這支示範兩件 MLOps 的事：

  1) **PSI（population stability index）抓漂移**：把「線上分布 vs 訓練分布」量化成一個數字，
     用業界門檻判讀（<0.1 穩定、0.1–0.25 留意、>0.25 顯著漂移→建議重訓）。
  2) **shadow 一致率**：候選模型在「影子」裡跟著算同一批輸入，回報跟現行的一致率——
     並示範本章的坑：**低一致率不一定是壞事**（更好的模型本來就該跟舊的不一樣）。

用法：
    python tiny_drift.py        # 純 CPU、瞬間、不需語料
"""

import math


def histogram(xs, edges):
    """把 xs 分箱成機率（Laplace 平滑：每箱 +0.5，避免空箱讓 log 比值爆掉）。"""
    nb = len(edges) - 1
    counts = [0] * nb
    for x in xs:
        for b in range(nb):
            if edges[b] <= x < edges[b + 1]:
                counts[b] += 1
                break
    tot = sum(counts)
    return [(c + 0.5) / (tot + 0.5 * nb) for c in counts]


def psi(ref, new, edges):
    """PSI = Σ (p_new - p_ref) · ln(p_new / p_ref)。0=同分布，越大越漂。"""
    p_ref, p_new = histogram(ref, edges), histogram(new, edges)
    return sum((pn - pr) * math.log(pn / pr) for pr, pn in zip(p_ref, p_new))


def classify(v):
    return "穩定（不必動）" if v < 0.1 else \
           "留意（持續監控）" if v < 0.25 else "顯著漂移 → 建議重訓"


def make_rng(seed=7):
    state = [seed]

    def rand():                                  # LCG → [0,1)，可重現
        state[0] = (1103515245 * state[0] + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF
    return rand


def sample_lengths(rand, n, mean, sigma):
    """模擬「請求的 token 長度」分布：近似常態 N(mean, sigma)（中央極限定理）。"""
    out = []
    for _ in range(n):
        z = sum(rand() for _ in range(12)) - 6         # 12 個 U(0,1) 相加 ≈ N(0,1)
        out.append(max(1.0, mean + z * sigma))
    return out


if __name__ == "__main__":
    rand = make_rng()
    edges = list(range(0, 61, 5)) + [1e9]              # 0,5,10,...,60,∞

    # 訓練分布：請求長度 ≈ N(25, 6)
    train = sample_lengths(rand, 5000, mean=25, sigma=6)
    scenarios = {
        "同分布（健康）":       sample_lengths(rand, 2000, mean=25, sigma=6),
        "輕微漂移（變長一點）": sample_lengths(rand, 2000, mean=28, sigma=6),
        "嚴重漂移（換了流量）": sample_lengths(rand, 2000, mean=32, sigma=7),
    }

    print("=== drift 監控：PSI（線上請求長度 vs 訓練分布）===")
    print(f"{'情境':<22}{'PSI':>8}   判讀")
    print("-" * 52)
    for name, online in scenarios.items():
        v = psi(train, online, edges)
        print(f"{name:<22}{v:>8.3f}   {classify(v)}")
    print("\n→ 一個數字就把「線上有沒有偏離訓練」量化、可設門檻自動觸發重訓。\n")

    # shadow 比對：候選 vs 現行在同一批輸入上的一致率（模擬 greedy 輸出）
    print("=== shadow 一致率（候選 vs 現行，同一批輸入）===")
    rng2 = make_rng(99)
    n = 1000
    # 候選有 ~30% 機率給出不同（且更好）的答案
    same = sum(1 for _ in range(n) if rng2() > 0.30)
    print(f"  一致率：{same}/{n} = {same/n*100:.1f}%")
    print("  低一致率不一定是壞事——更好的模型本來就該跟舊的不一樣。")
    print("  放行的硬條件是『品質 gate（更好就放行）』，一致率只告訴你『改動多大、要多謹慎驗』。")
    print("  把一致率當硬門檻，會擋掉每一次真進步——這又是一個『選對指標』。")
