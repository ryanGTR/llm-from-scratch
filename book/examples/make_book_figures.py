"""make_book_figures.py — 從「真跑的範例」產生書裡的插圖（不杜撰、可重現）。

每張圖都由對應的 tiny_*.py 真跑出來的數字畫成，存進 ../images/。
座標軸/標題用英文（避免 matplotlib 缺中文字型出現方框）；中文解說放在 .qmd 的圖說。

用法（純 CPU，約 5 分鐘；需先有 input.txt）：
    python make_book_figures.py
"""

import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

IMG = "../images"


# ---- Ch3：KV-cache 加速隨生成長度張開（tiny_kvcache）----
def fig_kvcache():
    import tiny_kvcache as kv
    torch.manual_seed(0)
    model = kv.GPT().eval()
    lengths = [50, 100, 200, 400, 600, 800]
    naive_t, cached_t, speedup = [], [], []
    prompt = torch.randint(0, kv.vocab_size, (1, 8))
    for n in lengths:
        t0 = time.perf_counter(); a = model.generate_naive(prompt, n); tn = time.perf_counter() - t0
        t0 = time.perf_counter(); b = model.generate_cached(prompt, n); tc = time.perf_counter() - t0
        assert torch.equal(a, b), "cached 必須與 naive 逐 token 相同"
        naive_t.append(tn); cached_t.append(tc); speedup.append(tn / tc)
        print(f"  n={n:>4}  naive {tn:5.2f}s  cached {tc:5.2f}s  {tn/tc:4.1f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(lengths, naive_t, "o-", color="#c0392b", label="naive  (recompute, O(T²))")
    ax1.plot(lengths, cached_t, "o-", color="#27ae60", label="KV-cache (O(T))")
    ax1.set_xlabel("tokens generated"); ax1.set_ylabel("wall-clock (s)")
    ax1.set_title("Generation time"); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(lengths, speedup, "o-", color="#2c3e50")
    ax2.set_xlabel("tokens generated"); ax2.set_ylabel("speedup  (naive / cached)")
    ax2.set_title("KV-cache speedup grows with length"); ax2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{IMG}/kvcache_speedup.png", dpi=120)
    print("  -> kvcache_speedup.png")


# ---- Ch4：聚合指標說健康，偵測器才現形 + LSH 候選暴減（tiny_dedup）----
def fig_dedup():
    import tiny_dedup as dd
    rand = dd.make_rand()
    text = open("input.txt", encoding="utf-8").read()
    docs, dup_truth = dd.build_corpus(text, rand)
    N = len(docs)
    hit_pct = len(dd.detector_hits(docs)) / N * 100
    r = dd.dedup(docs, rand)
    print(f"  detector hit {hit_pct:.1f}%  candidates {len(r['candidates'])} / "
          f"naive {r['naive_pairs']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    # (a) 聚合指標都「過」(0% 旗標) vs 偵測器抓到 20.7%
    bars = ax1.bar(["avg length\n(aggregate)", "char entropy\n(aggregate)", "markup detector\n(per-doc rule)"],
                   [0, 0, hit_pct], color=["#27ae60", "#27ae60", "#c0392b"])
    ax1.set_ylabel("% docs flagged as dirty")
    ax1.set_title("Aggregate says healthy — detector disagrees")
    ax1.bar_label(bars, fmt="%.1f%%", padding=3)
    ax1.set_ylim(0, hit_pct * 1.3)
    # (b) 候選對數量 naive vs LSH（log）
    b2 = ax2.bar(["naive\nC(N,2)", "LSH\ncandidates"],
                 [r["naive_pairs"], len(r["candidates"])], color=["#c0392b", "#27ae60"])
    ax2.set_yscale("log"); ax2.set_ylabel("pairs compared (log)")
    ax2.set_title(f"LSH compares {len(r['candidates'])/r['naive_pairs']*100:.1f}% of the pairs")
    ax2.bar_label(b2, fmt="%d", padding=3)
    fig.tight_layout(); fig.savefig(f"{IMG}/dedup_blindspot.png", dpi=120)
    print("  -> dedup_blindspot.png")


# ---- Ch7：train-acc 衝 100%，held-out 才說真話（tiny_dpo）----
def fig_dpo():
    import tiny_dpo as dp
    torch.manual_seed(1337)
    base = dp.pretrain()
    results = {}
    for kind in ("learnable", "random"):
        tr, ho = dp.run_dpo(base, kind)
        results[kind] = (tr * 100, ho * 100)
        print(f"  {kind:<10} train {tr*100:5.1f}%  held-out {ho*100:5.1f}%")

    labels = ["A: learnable\n(real vs scrambled)", "B: unlearnable\n(random label)"]
    train = [results["learnable"][0], results["random"][0]]
    held = [results["learnable"][1], results["random"][1]]
    x = range(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    b1 = ax.bar([i - w/2 for i in x], train, w, label="train-acc", color="#95a5a6")
    b2 = ax.bar([i + w/2 for i in x], held, w, label="held-out-acc", color="#2c3e50")
    ax.axhline(50, ls="--", color="#c0392b", lw=1)
    ax.text(len(labels) - 0.5, 52, "chance (50%)", color="#c0392b", fontsize=8, ha="right")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("preference accuracy (%)"); ax.set_ylim(0, 112)
    ax.set_title("DPO: train-acc lies, held-out tells the truth")
    ax.bar_label(b1, fmt="%.0f", padding=2); ax.bar_label(b2, fmt="%.0f", padding=2)
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(f"{IMG}/dpo_trainvsheldout.png", dpi=120)
    print("  -> dpo_trainvsheldout.png")


# ---- Ch6：promotion gate 怎麼擋（tiny_serve）----
def fig_gate():
    import tiny_serve as sv
    current = sv.train(300, seed=1)
    candidate = sv.train(900, seed=2)
    cur_loss, cand_loss = sv.val_loss(current), sv.val_loss(candidate)
    print(f"  current {cur_loss:.3f}  candidate {cand_loss:.3f}")

    names = ["candidate\n(gate=False)", "candidate\n(gate=True)", "rogue\n(unregistered)"]
    losses = [cand_loss, cand_loss, cur_loss * 1.05]   # rogue：示意一顆沒註冊的
    colors = ["#7f8c8d", "#27ae60", "#c0392b"]
    decisions = ["BLOCK", "PROMOTE", "RED FLAG"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bars = ax.bar(names, losses, color=colors)
    ax.axhline(cur_loss, ls="--", color="#2c3e50", lw=1)
    ax.text(2.4, cur_loss, "current (bar to beat)", color="#2c3e50", fontsize=8,
            va="bottom", ha="right")
    for bar, dec in zip(bars, decisions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, dec,
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("val loss"); ax.set_ylim(0, max(losses) * 1.25)
    ax.set_title("Promotion gate: only a better, registered model passes")
    fig.tight_layout(); fig.savefig(f"{IMG}/governance_gate.png", dpi=120)
    print("  -> governance_gate.png")


# ---- Ch6：可觀測性——冷啟動 + 尾延遲（tiny_observability）----
def fig_latency():
    import tiny_observability as ob
    svc = ob.Service()
    prompt = torch.randint(0, ob.vocab_size, (1, 8))
    for _ in range(30):
        svc.generate(prompt)
    cold, warm = svc.latencies[0], svc.latencies[1:]
    p50, p95 = ob.pct(warm, 50), ob.pct(warm, 95)
    print(f"  cold {cold:.1f}  p50 {p50:.1f}  p95 {p95:.1f}")

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.bar(["cold start\n(1st request)", "warm p50", "warm p95"],
                  [cold, p50, p95], color=["#c0392b", "#27ae60", "#e67e22"])
    ax.bar_label(bars, fmt="%.1f ms", padding=3)
    ax.set_ylabel("latency (ms)"); ax.set_ylim(0, cold * 1.3)
    ax.set_title("Observability: cold start & tail latency you'd miss with an average")
    fig.tight_layout(); fig.savefig(f"{IMG}/serving_latency.png", dpi=120)
    print("  -> serving_latency.png")


# ---- Ch8：PSI 抓資料漂移（tiny_drift）----
def fig_drift():
    import tiny_drift as dr
    rand = dr.make_rng()
    edges = list(range(0, 61, 5)) + [1e9]
    train = dr.sample_lengths(rand, 5000, 25, 6)
    scen = [("in-dist\n(healthy)", dr.sample_lengths(rand, 2000, 25, 6)),
            ("mild drift", dr.sample_lengths(rand, 2000, 28, 6)),
            ("severe drift", dr.sample_lengths(rand, 2000, 32, 7))]
    vals = [dr.psi(train, s, edges) for _, s in scen]
    print(f"  PSI: {[round(v,3) for v in vals]}")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = ["#27ae60" if v < 0.1 else "#e67e22" if v < 0.25 else "#c0392b" for v in vals]
    bars = ax.bar([s for s, _ in scen], vals, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(0.1, ls="--", color="#7f8c8d", lw=1); ax.text(2.4, 0.11, "0.10 watch", fontsize=8, ha="right", color="#7f8c8d")
    ax.axhline(0.25, ls="--", color="#c0392b", lw=1); ax.text(2.4, 0.27, "0.25 retrain", fontsize=8, ha="right", color="#c0392b")
    ax.set_ylabel("PSI (online vs training)")
    ax.set_title("Data drift: one number triggers retraining")
    fig.tight_layout(); fig.savefig(f"{IMG}/drift_psi.png", dpi=120)
    print("  -> drift_psi.png")


# ---- Ch5：先贏過笨基準（tiny_baseline）----
def fig_baseline():
    import tiny_baseline as tb
    rows = tb.results()
    print("  " + "  ".join(f"{n}={b:.2f}" for n, b in rows))
    names = [n for n, _ in rows]
    vals = [b for _, b in rows]
    colors = ["#95a5a6", "#e67e22", "#7f8c8d", "#27ae60"]   # GPT 綠
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(names, vals, color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("BPC (bits/char)  —  lower is better")
    ax.set_title("Earn your complexity: tiny GPT vs dumb baselines (same val)")
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout(); fig.savefig(f"{IMG}/baseline_bpc.png", dpi=120)
    print("  -> baseline_bpc.png")


if __name__ == "__main__":
    print("Ch5 baseline:");      fig_baseline()
    print("Ch3 kvcache:");       fig_kvcache()
    print("Ch4 dedup:");         fig_dedup()
    print("Ch6 latency:");       fig_latency()
    print("Ch6 governance:");    fig_gate()
    print("Ch8 drift:");         fig_drift()
    print("Ch7 dpo (~3min):");   fig_dpo()
    print("done.")
