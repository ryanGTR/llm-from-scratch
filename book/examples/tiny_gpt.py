"""tiny_gpt.py — 一支自包含、CPU 就能跑的最小 char-level GPT。

書本 Ch1 的「💻 在你的機器上」配套程式。沒有任何專案相依，只要 torch。
拿 1.1 MB 的莎士比亞當語料，在筆電 CPU 上約一分鐘就能從「亂碼」訓練到
「長得像英文的假莎士比亞」——親眼看到 next-token prediction 在動。

用法：
    # 1) 抓語料（任選其一）
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    # 2) 跑（純 CPU，不需 GPU）
    python tiny_gpt.py

這支刻意用「樸素版 attention」（把整個 T×T 矩陣攤出來），對照書裡的數學式，
不是為了快，是為了看得懂。正式專案版在 src/model.py。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- 超參數（小到 CPU 也跑得動，但已足以學到結構）----
block_size = 64        # context length：一次最多看 64 個字
n_embd     = 128       # embedding 維度
n_head     = 4         # attention 頭數（128 / 4 = 每頭 32 維）
n_layer    = 3         # 疊 3 個 Transformer block
max_iters  = 3000      # 訓練步數
eval_every = 500
batch_size = 32
lr         = 3e-3
torch.manual_seed(1337)

# ---- 資料：char-level，把每個字元映成一個整數 ----
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}      # char -> id
itos = {i: c for i, c in enumerate(chars)}      # id  -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    """隨機切出 batch_size 條長度 block_size 的序列；y 是 x 整體右移一格。"""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # 目標＝下一個字
    return x, y


# ---- 模型：token+pos embedding → N×block → 線性輸出每個字的機率 ----
class Head(nn.Module):
    """單一 attention head：書裡那條 softmax(QKᵀ/√d)V 的直接翻譯。"""

    def __init__(self, head_dim):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_dim, bias=False)
        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)
        # 下三角遮罩：第 i 個位置只能看 <= i（不能偷看未來）
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        att = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])    # (B,T,T) 相似度
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # 因果遮罩
        att = F.softmax(att, dim=-1)                              # 轉成權重
        return att @ v                                           # 加權平均 value


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        hd = n_embd // n_head
        self.heads = nn.ModuleList([Head(hd) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # 多頭併回去
        return self.proj(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHead()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual：x = x + f(x)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        x = self.head(self.lnf(self.blocks(x)))               # (B,T,vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, vocab_size), targets.view(-1))
        return x, loss

    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits, _ = self(idx[:, -block_size:])            # 只看最後 block_size 個字
            probs = F.softmax(logits[:, -1, :], dim=-1)        # 最後一格的下一字分佈
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.stack([model(*get_batch(split))[1] for _ in range(50)])
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    model = TinyGPT()
    print(f"參數量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M，"
          f"vocab={vocab_size}，device=cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for it in range(max_iters + 1):
        if it % eval_every == 0:
            l = estimate_loss(model)
            print(f"step {it:>4}: train {l['train']:.3f}  val {l['val']:.3f}")
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print("\n--- 生成樣本（從換行字元起手）---")
    start = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(start, 400)[0].tolist()))
