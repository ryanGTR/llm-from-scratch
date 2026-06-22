"""tiny_eval.py — 在 CPU 上親手做三件評估紀律：亂猜基準、BPC、多 seed 重大性。

書本 Ch5 的「💻 在你的機器上」配套程式。評估難的不是算數字，是**選對該量的東西**。
這支用 Ch1 的 char-level 小 GPT 把三條紀律跑一遍、印出真數字：

  1) **先有亂猜基準**：vocab=V 的模型若完全亂猜，loss = ln V。先量基準，val loss 才有意義。
  2) **用對的尺（BPC）**：raw cross-entropy 的尺度隨 vocab 變，跨 tokenizer 不能比；
     換算成 bits/char（BPC）才公平。char-level 下 BPC = loss / ln2。
  3) **多 seed 看重大性**：單次裸數字帶運氣。跑 N 個 seed 報 mean ± std，
     才知道一個差異是真的、還是雜訊。

用法：
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python tiny_eval.py           # 純 CPU，3 個 seed 約 3–4 分鐘
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 64
n_embd     = 128
n_head     = 4
n_layer    = 3
max_iters  = 1500
batch_size = 32
seeds      = (1337, 42, 7)

text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.key = nn.Linear(n_embd, hd, bias=False)
        self.query = nn.Linear(n_embd, hd, bias=False)
        self.value = nn.Linear(n_embd, hd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        att = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        att = att.masked_fill(self.tril[:x.shape[1], :x.shape[1]] == 0, float("-inf"))
        return F.softmax(att, dim=-1) @ v


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        hd = n_embd // n_head
        self.heads = nn.ModuleList([Head(hd) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.proj(torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        return self.head(self.lnf(self.blocks(x)))


@torch.no_grad()
def val_loss(model):
    model.eval()
    l = torch.stack([
        F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        for x, y in (get_batch("val") for _ in range(50))]).mean().item()
    model.train()
    return l


def train_one(seed):
    torch.manual_seed(seed)
    model = GPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(max_iters):
        x, y = get_batch("train")
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return val_loss(model)


if __name__ == "__main__":
    baseline = math.log(vocab_size)
    print(f"vocab={vocab_size}  device=cpu")
    print(f"亂猜基準 = ln {vocab_size} = {baseline:.3f} nat（模型要明顯低於它才算在學）\n")

    losses = []
    for s in seeds:
        vl = train_one(s)
        losses.append(vl)
        print(f"  seed {s:>4}: val loss {vl:.4f}   BPC {vl/math.log(2):.3f} bits/char")

    t = torch.tensor(losses)
    mean, std = t.mean().item(), t.std().item()
    print(f"\n{len(seeds)} seeds: val loss {mean:.4f} ± {std:.4f}"
          f"   BPC {mean/math.log(2):.3f} ± {std/math.log(2):.3f} bits/char")
    print(f"相對亂猜基準 {baseline:.3f}，模型把不確定性砍了 "
          f"{(1 - mean/baseline)*100:.0f}%。")
    print(f"\n重大性：std≈{std:.3f}。一個架構差異要比這個尺度大幾倍，才能宣稱『真的有差』、"
          f"\n而不是 seed 的運氣——這就是要不要多跑 seed 的判準。")
