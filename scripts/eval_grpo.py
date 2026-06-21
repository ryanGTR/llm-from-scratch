"""GRPO 評估：把「有 KL 錨 vs 無 KL 錨」兩條訓練曲線並排，揭露 reward hacking。

RLHF 的核心張力：RM 是「學來的代理指標」，RL 在拚命最大化它。如果沒有 KL 錨把 policy
拉住，最大化代理常常會犧牲真實品質——policy 找到鑽 RM 漏洞的捷徑（mode collapse／退化），
RM 分數漂亮、人看了卻是爛的。這就是 Goodhart：指標一旦變成目標，就不再是好指標。

兩把尺：
- 代理＝RM 分數（RL 在最大化）
- 真實＝生成多樣性 distinct-output（RM 沒直接管）+ 重複率

  python scripts/eval_grpo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

RUNS = [   # (標籤, csv, 顏色)
    ("KL 錨 ON (β=0.05)", "grpo.csv", "#2ca02c"),
    ("KL 錨 OFF (β=0)", "grpo_hack.csv", "#d62728"),
]


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

    ax1.set_xlabel("GRPO 步數"); ax1.set_ylabel("RM 分數（代理指標，RL 在最大化）")
    ax1.set_title("代理指標：RM 分數")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    ax2.set_xlabel("GRPO 步數"); ax2.set_ylabel("生成多樣性 %（真實品質，RM 沒管）")
    ax2.set_title("真實品質：輸出多樣性（崩=mode collapse=reward hacking）")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    out = ART / "grpo_reward_hacking.png"
    fig.suptitle("RLHF 招牌坑 reward hacking：代理漲、真實崩（Goodhart）", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"圖 → {out}")
    for label, csv, _ in RUNS:
        p = ART / "runs" / csv
        if not p.exists():
            continue
        d = read(p)
        print(f"  [{label}] RM 分數 {d['reward'][0]:+.2f}→{d['reward'][-1]:+.2f} | "
              f"多樣性 {d['diversity'][0]*100:.0f}%→{d['diversity'][-1]*100:.0f}% | "
              f"重複率 {d['repetition'][0]*100:.0f}%→{d['repetition'][-1]*100:.0f}%")


if __name__ == "__main__":
    main()
