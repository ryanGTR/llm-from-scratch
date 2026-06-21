"""DPO 專用評估：在「沒訓過的 held-out 偏好對」上比 SFT(reference) vs DPO(policy)。

延續 SFT 評估的教訓「選對指標、質疑指標被汙染」，這裡同時呈現三件事：

1. 量「偏好準確率」要用「每 token 平均 logπ」不是「總和」——總和有長度偏誤（答案越長
   越吃虧），會把「品質」和「長度」混在一起。
2. 兩種偏好軸的對照（這是本里程碑的核心發現）：
     format（連貫 vs 退化重複）＝模型容量內 → DPO 在 held-out 真的類推（acc 大漲）
     topic （on-topic vs 張冠李戴）＝超出 8M 容量 → DPO 只背 train、held-out 學不動
3. 過度優化：train-acc 會衝到 100%，但 held-out 才是真話 → 訓練曲線圖一看便知。

  python scripts/eval_dpo.py
"""

import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig          # noqa: E402
from src.model import GPT                  # noqa: E402
from src.tokenizer import load_tokenizer   # noqa: E402

dpo = importlib.import_module("pipeline.06_dpo")   # 模組名以數字開頭，用 import_module
ART = ROOT / "artifacts"

MODES = [   # (名稱, ckpt, held-out, run csv, 說明)
    ("format（連貫 vs 退化）", "dpo_format_ckpt.pt", "dpo_format_heldout.jsonl",
     "dpo_format.csv", "容量內 → 會類推"),
    ("topic （對題 vs 張冠李戴）", "dpo_ckpt.pt", "dpo_heldout.jsonl",
     "dpo.csv", "超出容量 → 只會背"),
]


def load(p, device):
    ck = torch.load(p, map_location=device)
    m = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def norm_pref_acc(model, tok, rows, device):
    """held-out 偏好準確率（每 token 平均 logπ：chosen > rejected 的比例，去長度偏誤）。"""
    wins = 0
    for r in rows:
        means = []
        for key in ("chosen", "rejected"):
            ids, m = dpo.build_example(tok, r["prompt"], r[key], 128)
            X = torch.tensor([ids], device=device)
            M = torch.tensor([m], dtype=torch.float, device=device)
            ntok = max(1.0, M[:, 1:].sum().item())
            means.append(dpo.seq_logp(model, X, M).item() / ntok)
        wins += int(means[0] > means[1])
    return wins / len(rows)


def read_csv(path):
    lines = Path(path).read_text().strip().splitlines()
    cols = lines[0].split(",")
    rows = [dict(zip(cols, ln.split(","))) for ln in lines[1:]]
    step = [int(r["step"]) for r in rows]
    held = [float(r["held_acc"]) * 100 for r in rows]
    train = [float(r["train_acc"]) * 100 for r in rows]
    return step, train, held


def make_plot(device, tok):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"format": "#2ca02c", "topic": "#d62728"}
    for (name, _ck, _ho, csv, _note) in MODES:
        p = ART / "runs" / csv
        if not p.exists():
            continue
        key = "format" if "format" in csv else "topic"
        step, train, held = read_csv(p)
        ax.plot(step, held, "-o", color=colors[key], label=f"{name}：held-out", ms=4)
        ax.plot(step, train, "--", color=colors[key], alpha=0.35,
                label=f"{name}：train")
    ax.axhline(50, ls=":", color="gray", lw=1)
    ax.text(5, 52, "50% = 隨機猜", color="gray", fontsize=9)
    ax.set_xlabel("DPO 訓練步數")
    ax.set_ylabel("偏好準確率 (%)")
    ax.set_title("DPO 類推 vs 死背：train-acc 都衝 100%，held-out 才說真話")
    ax.set_ylim(-3, 103)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.2)
    out = ART / "dpo_generalization.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\n訓練曲線圖 → {out}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(ART / "tokenizer.json")
    ref = load(ART / "sft_ckpt.pt", device)        # 兩種模式共用的起點 = SFT

    print("=" * 64)
    print("held-out 偏好準確率（每 token 平均 logπ；chosen 勝率）— SFT vs DPO")
    print("=" * 64)
    for (name, ck, ho, _csv, note) in MODES:
        ckpt = ART / ck
        held_p = ART / ho
        if not ckpt.exists() or not held_p.exists():
            print(f"  [{name}] 缺檔，先跑 make dpo")
            continue
        held = [__import__("json").loads(l) for l in
                held_p.read_text(encoding="utf-8").splitlines()]
        pol = load(ckpt, device)
        r_acc = norm_pref_acc(ref, tok, held, device)
        p_acc = norm_pref_acc(pol, tok, held, device)
        print(f"  {name:22s} SFT {r_acc*100:4.0f}% → DPO {p_acc*100:4.0f}%  "
              f"({(p_acc-r_acc)*100:+.0f} pt)  {note}")
    print("=" * 64)
    print("解讀：DPO 兩種模式的 train-acc 都會到 100%；唯一差別在 held-out 能不能類推。")
    print("      format 真學到可遷移的『避免退化』特徵；topic 需要語義綁定、8M 學不動只能背。")

    try:
        make_plot(device, tok)
    except Exception as e:                          # 無 matplotlib 不該讓評估失敗
        print(f"（略過畫圖：{e}）")


if __name__ == "__main__":
    main()
