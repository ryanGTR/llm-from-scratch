"""PPO 評估：clip vs 無 clip 對照，展示 PPO 招牌零件「截斷」防止破壞性更新。

PPO 用同一批 rollout 訓好幾個 epoch（省取樣），但這樣 policy 容易一步走太遠把自己走壞。
clipped surrogate 把 importance ratio 夾在 [1-ε,1+ε] → 限制單次更新幅度＝「Proximal（近端）」。
為了**隔離 clip 的作用**，這個對照刻意關掉 KL 罰（β=0）、用較大 lr 與 8 epochs。

兩張圖（真正的訊號是「崩沒崩」，不是 KL）：
- 左：RM 分數。clip 穩定爬升；**無 clip 衝一下就崩盤**（一個過大更新把 policy 走壞）。
- 右：生成多樣性。clip 維持 100%；**無 clip mode collapse 到個位數**＝policy 被那步更新毀了。

  python scripts/eval_ppo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

RUNS = [("clip ε=0.2", "ppo_clip.csv", "#2ca02c"), ("無 clip", "ppo_noclip.csv", "#d62728")]


def read(path):
    lines = Path(path).read_text().strip().splitlines()
    cols = lines[0].split(",")
    rows = [dict(zip(cols, ln.split(","))) for ln in lines[1:]]
    return {c: [float(r[c]) for r in rows] for c in cols}


def main():
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, csv, color in RUNS:
        p = ART / "runs" / csv
        if not p.exists():
            continue
        d = read(p)
        ax1.plot(d["step"], d["reward"], "-o", color=color, label=label, ms=4)
        ax2.plot(d["step"], [v * 100 for v in d["diversity"]], "-o", color=color, label=label, ms=4)

    ax1.set_xlabel("PPO 步數"); ax1.set_ylabel("RM 分數")
    ax1.set_title("無 clip：衝一下就崩盤；clip：穩定爬升")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    ax2.set_xlabel("PPO 步數"); ax2.set_ylabel("生成多樣性 %")
    ax2.set_title("無 clip：mode collapse（policy 被一步走壞）；clip：維持 100%")
    ax2.set_ylim(-3, 103); ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    fig.suptitle("PPO 的 clip＝防破壞性更新：一步走太遠會把 policy 走壞（β=0 隔離 clip）", fontsize=12)
    out = ART / "ppo_clip_vs_noclip.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"圖 → {out}")
    for label, csv, _ in RUNS:
        p = ART / "runs" / csv
        if p.exists():
            d = read(p)
            print(f"  [{label}] RM {d['reward'][0]:+.1f}→{d['reward'][-1]:+.1f} | "
                  f"多樣性 {d['diversity'][0]*100:.0f}%→{d['diversity'][-1]*100:.0f}% | "
                  f"clip% {d['clip_frac'][-1]*100:.0f} | value_loss→{d['value_loss'][-1]:.2f}")


if __name__ == "__main__":
    main()
