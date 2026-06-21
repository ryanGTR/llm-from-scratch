"""DPO vs IPO 對照：IPO 把 margin 釘在固定目標 1/(2β)，而非讓它爆掉。

接 derivations §3（DPO margin 尺度≈1/β、σ 飽和後仍緩慢推爆）與 dpo-beta（過度優化）。
IPO（Azar 2023）用平方損失 (margin − 1/(2β))² 取代 −logσ(β·margin)，給 margin 一個**有限目標**：

- DPO：margin 一路衝（實測 200+），σ 飽和也停不下來 → 過度優化。
- IPO：margin 收斂到 1/(2β) 附近就停 → 天生防過度優化；target 是顯式旋鈕。

讀 `artifacts/runs/cmp_dpo.csv` 與 `cmp_ipo.csv`（同資料、同 β=0.1、同步數，只差損失函數）出圖。

  python scripts/dpo_vs_ipo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
BETA = 0.1
TARGET = 1.0 / (2 * BETA)        # IPO 的 margin 目標 = 5


def read(path):
    lines = Path(path).read_text().strip().splitlines()
    cols = lines[0].split(",")
    rows = [dict(zip(cols, ln.split(","))) for ln in lines[1:]]
    return {c: [float(r[c]) for r in rows] for c in cols}


def main():
    import matplotlib.pyplot as plt
    d = read(ART / "runs" / "cmp_dpo.csv")
    i = read(ART / "runs" / "cmp_ipo.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.plot(d["step"], d["margin"], "-o", color="#d62728", label="DPO（−logσ(β·m)）", ms=4)
    ax1.plot(i["step"], i["margin"], "-o", color="#1f77b4", label="IPO（(m−1/2β)²）", ms=4)
    ax1.axhline(TARGET, ls=":", color="#1f77b4", alpha=0.7)
    ax1.text(20, TARGET + 6, f"IPO 目標 1/(2β)={TARGET:.0f}", color="#1f77b4", fontsize=9)
    ax1.set_xlabel("步數"); ax1.set_ylabel("margin（log-ratio 差）")
    ax1.set_title("margin：DPO 爆衝 vs IPO 釘在目標")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    ax2.plot(d["step"], [v * 100 for v in d["held_acc"]], "-o", color="#d62728", label="DPO", ms=4)
    ax2.plot(i["step"], [v * 100 for v in i["held_acc"]], "-o", color="#1f77b4", label="IPO", ms=4)
    ax2.set_xlabel("步數"); ax2.set_ylabel("held-out 偏好準確率 (%)")
    ax2.set_title("held-out：clean 偏好下 DPO 的激進無害故較高")
    ax2.set_ylim(50, 102); ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    fig.suptitle("DPO vs IPO：IPO 用有限目標換「不過度優化」（margin 受控）", fontsize=12)
    out = ART / "dpo_vs_ipo.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"圖 → {out}")
    print(f"  DPO : margin {d['margin'][1]:.0f}→{d['margin'][-1]:.0f}（爆）| held-out {d['held_acc'][-1]*100:.0f}%")
    print(f"  IPO : margin 釘在 ~{TARGET:.0f}（{i['margin'][-1]:.0f}）| held-out {i['held_acc'][-1]*100:.0f}%")
    print("  IPO target 旋鈕（β 越小目標越大、偏好越強）：")
    for b, csv in [(0.1, "cmp_ipo"), (0.05, "ipo_b0.05"), (0.02, "ipo_b0.02")]:
        p = ART / "runs" / f"{csv}.csv"
        if p.exists():
            r = read(p)
            print(f"    β={b} target={1/(2*b):.0f} → margin {r['margin'][-1]:.0f}, held-out {r['held_acc'][-1]*100:.0f}%")


if __name__ == "__main__":
    main()
