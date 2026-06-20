"""Deep Ensemble：把多個「不同 seed 訓練」的模型「機率平均」起來當一個 ensemble。

這是隨機森林的神經網路版——回收 B④ 多 seed 的概念，但這次「留下模型、平均預測」
而不是只看統計。理論：獨立訓練的模型各犯不同的錯，平均後錯誤互相抵消 → loss 更低、
更穩。看實測是否成立。

用法：python scripts/deep_ensemble.py
"""

import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig, TrainConfig   # noqa: E402
from src.model import GPT                        # noqa: E402

ART = ROOT / "artifacts"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ITERS, SEEDS, BLOCK, BATCH = 2500, [0, 1, 2], 128, 32

vocab = __import__("json").loads((ART / "meta.json").read_text())["vocab_size"]
train_data = np.fromfile(ART / "train.bin", dtype=np.uint16)
val_data = np.fromfile(ART / "val.bin", dtype=np.uint16)


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK, (BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(DEV), y.to(DEV)


def train_one(seed):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=vocab, n_layer=4, n_head=4, n_embd=128, block_size=BLOCK)
    m = GPT(cfg).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1)
    for _ in range(ITERS):
        x, y = get_batch(train_data)
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


# 固定一組 val batch，讓「單模型」與「ensemble」用同樣的考題（公平）
torch.manual_seed(999)
VAL_BATCHES = [get_batch(val_data) for _ in range(100)]


@torch.no_grad()
def single_loss(m):
    return st.mean(F.cross_entropy(m(x)[0].reshape(-1, vocab), y.reshape(-1)).item()
                   for x, y in VAL_BATCHES)


@torch.no_grad()
def ensemble_loss(models):
    losses = []
    for x, y in VAL_BATCHES:
        probs = sum(F.softmax(m(x)[0], dim=-1) for m in models) / len(models)
        p_true = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        losses.append((-torch.log(p_true + 1e-9)).mean().item())
    return st.mean(losses)


print(f"訓練 {len(SEEDS)} 個模型（seed {SEEDS}）...")
models = [train_one(s) for s in SEEDS]
singles = [single_loss(m) for m in models]
ens = ensemble_loss(models)

print("\n" + "=" * 44)
for s, v in zip(SEEDS, singles):
    print(f"  單模型 seed={s}：val loss {v:.4f}")
print(f"  單模型平均：{st.mean(singles):.4f}（最佳單一 {min(singles):.4f}）")
print(f"  ★ Ensemble（3 個平均）：{ens:.4f}")
print(f"  ensemble vs 最佳單一：{min(singles) - ens:+.4f}")

# 圖
plt.figure(figsize=(7.5, 4.3))
labels = [f"seed {s}" for s in SEEDS] + ["Ensemble"]
vals = singles + [ens]
colors = ["#4c72b0"] * len(SEEDS) + ["#c44e52"]
plt.bar(labels, vals, color=colors)
for i, v in enumerate(vals):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
plt.axhline(min(singles), ls="--", color="gray", alpha=0.6, label="最佳單一")
plt.ylim(min(vals) - 0.03, max(vals) + 0.02)
plt.ylabel("val loss（越低越好）")
plt.title("Deep Ensemble：3 個模型機率平均 → 比任何單一都低")
plt.legend(); plt.tight_layout()
plt.savefig(ART / "deep_ensemble.png", dpi=120)
print(f"\n已存：{ART}/deep_ensemble.png")
