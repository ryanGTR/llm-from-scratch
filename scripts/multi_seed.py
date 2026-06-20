"""B④ 多 seed 嚴謹：每個設定跑 N 個 seed，算 mean±std，判斷差異是真是雜訊。

核心觀念：單次跑分不出「真差異」還是「運氣」。跑多個 seed 看「自己跟自己的波動」
（std=雜訊地板），再比「設定間的差距」是否 > 約 2×std → 真差異；落在 std 內 → 平手。

用法：python scripts/multi_seed.py            # 預設比 classic/swiglu/rope，3 seeds
產物：artifacts/multi_seed.png（誤差線圖）+ 終端機印 mean±std 與判定。
"""

import csv
import statistics as stats
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ITERS, SEEDS = 3000, [0, 1, 2]
# 要坐實的設定：名稱 -> 額外的 train flags
CONFIGS = {
    "classic": [],
    "swiglu": ["--use_swiglu"],
    "rope": ["--use_rope"],
}


def best_val(run_name: str) -> float:
    with open(ROOT / "artifacts" / "runs" / f"{run_name}.csv", newline="") as f:
        return min(float(r["val_loss"]) for r in csv.DictReader(f))


def main():
    results: dict[str, list[float]] = {}
    for name, flags in CONFIGS.items():
        vals = []
        for s in SEEDS:
            run = f"ms_{name}_s{s}"
            print(f"  跑 {name} seed={s} ...", flush=True)
            subprocess.run(
                ["python", "pipeline/02_train.py", "--max_iters", str(ITERS),
                 "--block_size", "128", "--run_name", run, "--seed", str(s)] + flags,
                cwd=ROOT, capture_output=True)
            vals.append(best_val(run))
        results[name] = vals

    # mean / std
    print("\n" + "=" * 52)
    print(f"{'設定':>10} {'mean':>8} {'±std':>8}   每次")
    summ = {}
    for name, vals in results.items():
        m, sd = stats.mean(vals), (stats.stdev(vals) if len(vals) > 1 else 0.0)
        summ[name] = (m, sd)
        print(f"{name:>10} {m:>8.4f} {sd:>8.4f}   {[round(v,4) for v in vals]}")

    # 判定：每個 modern vs classic，差距 vs 2×std
    base_m, base_sd = summ["classic"]
    print("-" * 52)
    for name in CONFIGS:
        if name == "classic":
            continue
        m, sd = summ[name]
        gap = base_m - m                      # 正=比 classic 好
        noise = 2 * max(sd, base_sd)          # 粗略雜訊門檻
        verdict = "真差異 ✅" if abs(gap) > noise else "落在雜訊內＝平手"
        print(f"{name:>10} vs classic：差 {gap:+.4f}，2×std≈{noise:.4f} → {verdict}")

    # 誤差線圖
    names = list(CONFIGS)
    means = [summ[n][0] for n in names]
    stds = [summ[n][1] for n in names]
    plt.figure(figsize=(7, 4.2))
    plt.bar(names, means, yerr=stds, capsize=8,
            color=["#4c72b0", "#dd8452", "#55a868"])
    for i, (m, sd) in enumerate(zip(means, stds)):
        plt.text(i, m, f"{m:.3f}\n±{sd:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("best val loss（越低越好）")
    plt.title(f"多 seed（{len(SEEDS)} 次）mean ± std：誤差棒分得開才算真差異")
    plt.tight_layout()
    out = ROOT / "artifacts" / "multi_seed.png"
    plt.savefig(out, dpi=120)
    print(f"\n已存：{out}")


if __name__ == "__main__":
    main()
