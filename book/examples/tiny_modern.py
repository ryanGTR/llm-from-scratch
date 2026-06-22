"""tiny_modern.py — 在 CPU 上親手驗證「準 vs 省」：RMSNorm / SwiGLU / RoPE。

書本 Ch2 的「💻 在你的機器上」配套程式。延續 Ch1 的 char-level 小 GPT，
把三個現代零件做成可一鍵切換的開關，然後在同一份 1.1 MB 莎士比亞上跑對照，
印出 val loss 比較表——讓你親眼看到：哪個零件降 loss（準）、哪個只是更省（準度持平）。

用法：
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python tiny_modern.py            # 跑完整對照（baseline + 三個單項 + 全開）

注意：CPU 上每個配置約一兩分鐘，跑五個配置約 5–8 分鐘。這支用「省 epoch」的小設定，
deltas 會比 repo 全量訓練小一點、也帶一點雜訊，但「準/省」的方向會一致。
"""

import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

_SMOKE = bool(os.environ.get("BOOK_SMOKE"))   # CI 煙霧測試：極少步數只驗「跑得動」

block_size = 64
n_embd     = 128
n_head     = 4
n_layer    = 3
max_iters  = 30 if _SMOKE else 2500
eval_every = 2500          # 只在頭尾各評一次，省時間
batch_size = 32
lr         = 3e-3

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y


# ---- 現代零件（都可開關）----
class RMSNorm(nn.Module):                       # 「省」：砍掉減均值，只除 RMS
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight


def build_rope(head_dim, max_seq, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq).float(), inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):                     # 「準＋外推」：把 q/k 依位置旋轉
    T = x.shape[2]
    cos, sin = cos[:T].view(1, 1, T, -1), sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nh, self.hd, self.rope = n_head, n_embd // n_head, cfg["rope"]
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        if cfg["rope"]:
            cos, sin = build_rope(self.hd, block_size)
            self.register_buffer("cos", cos); self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        if self.rope:
            q, k = apply_rope(q, self.cos, self.sin), apply_rope(k, self.cos, self.sin)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class GELU_MLP(nn.Module):                        # 原版前饋層
    def __init__(self):
        super().__init__()
        self.fc, self.proj = nn.Linear(n_embd, 4 * n_embd), nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class SwiGLU(nn.Module):                           # 「準」：多一條閘門
    def __init__(self):
        super().__init__()
        h = int(round(8 / 3 * n_embd / 8) * 8)     # ≈8/3·n_embd，湊參數量公平
        self.g, self.u, self.d = (nn.Linear(n_embd, h), nn.Linear(n_embd, h),
                                  nn.Linear(h, n_embd))

    def forward(self, x):
        return self.d(F.silu(self.g(x)) * self.u(x))


def norm(cfg):
    return RMSNorm(n_embd) if cfg["rmsnorm"] else nn.LayerNorm(n_embd)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = norm(cfg), norm(cfg)
        self.attn = Attn(cfg)
        self.mlp = SwiGLU() if cfg["swiglu"] else GELU_MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = None if cfg["rope"] else nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(n_layer)])
        self.lnf = norm(cfg)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx)
        if self.pos is not None:                   # RoPE 模式不需要學習式位置表
            x = x + self.pos(torch.arange(T))
        x = self.head(self.lnf(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, vocab_size), targets.view(-1))
        return x, loss


@torch.no_grad()
def val_loss(model):
    model.eval()
    l = torch.stack([model(*get_batch("val"))[1] for _ in range(50)]).mean().item()
    model.train()
    return l


def train(cfg):
    torch.manual_seed(1337)                        # 同種子 → 公平對比
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(max_iters):
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return val_loss(model), sum(p.numel() for p in model.parameters())


OFF = {"rmsnorm": False, "swiglu": False, "rope": False}
CONFIGS = [
    ("baseline (LayerNorm+GELU+學習位置)", {**OFF}),
    ("+RMSNorm           (預測：省=持平)", {**OFF, "rmsnorm": True}),
    ("+SwiGLU            (預測：準=降)",   {**OFF, "swiglu": True}),
    ("+RoPE              (預測：準=降)",   {**OFF, "rope": True}),
    ("全開 (LLaMA 配方)",                   {"rmsnorm": True, "swiglu": True, "rope": True}),
]

if __name__ == "__main__":
    print(f"vocab={vocab_size}  {max_iters} 步  device=cpu\n")
    print(f"{'配置':<38}{'val loss':>10}{'參數':>10}")
    print("-" * 58)
    base = None
    for name, cfg in CONFIGS:
        vl, params = train(cfg)
        if base is None:
            base = vl
        delta = "" if vl == base else f"  ({vl - base:+.3f})"
        print(f"{name:<38}{vl:>10.4f}{params / 1e6:>9.2f}M{delta}")
