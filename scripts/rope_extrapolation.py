"""RoPE 招牌 demo：外推（train short, test long）。

兩個模型都用 block_size=256，但「只用長度 64 的序列訓練」（位置 64+ 從沒見過），
再測 64→256 各長度的 val loss：
  學習式位置 embedding：位置 64+ 是沒訓過的隨機向量 → loss 在 64 之後爆掉
  RoPE：位置用旋轉算出來、沒有「沒訓過」的問題 → 平順外推

用法：python scripts/rope_extrapolation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig, TrainConfig   # noqa: E402
from src.model import GPT                        # noqa: E402

ART = ROOT / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK, TRAIN_LEN, ITERS, BATCH = 256, 64, 2000, 32

vocab = json.loads((ART / "meta.json").read_text())["vocab_size"]
train_data = np.fromfile(ART / "train.bin", dtype=np.uint16)
val_data = np.fromfile(ART / "val.bin", dtype=np.uint16)


def get_batch(data, L):
    ix = torch.randint(len(data) - L, (BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + L].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + L].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def train(use_rope):
    torch.manual_seed(1337)
    cfg = GPTConfig(vocab_size=vocab, n_layer=4, n_head=4, n_embd=128,
                    block_size=BLOCK, use_rope=use_rope)
    m = GPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.1)
    for _ in range(ITERS):
        x, y = get_batch(train_data, TRAIN_LEN)   # 只用長度 64 訓練
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


@torch.no_grad()
def eval_at(m, L, n=50):
    m.eval()
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(val_data, L)
        tot += m(x, y)[1].item()
    m.eval()
    return tot / n


print(f"訓練長度={TRAIN_LEN}，測試長度到 {BLOCK}（device={DEVICE}）")
m_pos = train(False)
m_rope = train(True)
lengths = [64, 96, 128, 160, 192, 224, 256]
pos = [eval_at(m_pos, L) for L in lengths]
rope = [eval_at(m_rope, L) for L in lengths]

print(f"{'eval長度':>8} {'學習式位置':>10} {'RoPE':>8}")
for L, p, r in zip(lengths, pos, rope):
    print(f"{L:>8} {p:>10.3f} {r:>8.3f}")

plt.figure(figsize=(8, 4.5))
plt.axvline(TRAIN_LEN, color="gray", ls=":", label=f"訓練長度={TRAIN_LEN}")
plt.plot(lengths, pos, "o-", label="學習式位置 embedding")
plt.plot(lengths, rope, "o-", label="RoPE")
plt.xlabel("測試時的序列長度"); plt.ylabel("val loss（越低越好）")
plt.title("外推：訓練只看 64，測試餵更長 → RoPE 撐得住、學習式位置垮掉")
plt.legend(); plt.grid(alpha=0.3)
out = ART / "rope_extrapolation.png"
plt.tight_layout(); plt.savefig(out, dpi=120)
print(f"\n已存：{out}")
