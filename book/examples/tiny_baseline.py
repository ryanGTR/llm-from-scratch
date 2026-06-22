"""tiny_baseline.py — 在 CPU 上親手問：你的神經網路有沒有「賺到」它的複雜度？

書本 Ch5 的 baseline 對照：現在書裡的評估都跟自己比（val loss / BPC / 亂猜）。這支拿 tiny GPT
去跟兩個**笨但公認**的基準在**同一份 val、同一把尺（BPC, bits/char）**上比：

  - 亂猜：log2(vocab)，完全沒學的下限。
  - gzip：通用壓縮軟體的壓縮率＝資訊理論的天然 BPC（「你贏得過 gzip 嗎？」）。
  - n-gram（backoff + add-k）：char-level LM 的教科書基準（只數頻率、沒有神經網路）。
  - tiny GPT：本書的小 GPT。

重點不是打贏 GPT-2（8M 玩具不可能、也沒意義），是證明「神經網路真的學到 n-gram/gzip 學不到的
東西、值得這個複雜度」——而且**選下界基準而非虛榮基準，本身就是一個『選對 baseline』的示範**。

用法：
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python tiny_baseline.py        # 純 CPU，約 2 分鐘

（純函式 bpc_random / bpc_gzip / build_ngram / bpc_ngram 都吃參數、不依賴語料，方便 import 測試。）
"""

import gzip
import math
import os
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.nn import functional as F

_SMOKE = bool(os.environ.get("BOOK_SMOKE"))

block_size = 64
n_embd, n_head, n_layer = 128, 4, 3
max_iters = 30 if _SMOKE else 3000
batch_size = 32
NGRAM_ORDER = 4          # 4-gram，給 n-gram 一個公平（不弱）的版本
ADD_K = 0.05
LN2 = math.log(2)
torch.manual_seed(1337)


# ---------- 語料（延遲載入，import 時不碰檔案）----------
def load_corpus(path="input.txt"):
    global text, chars, V, stoi, data, train_ids, val_ids, train_text, val_text
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_ids, val_ids = data[:n], data[n:]
    train_text, val_text = text[:n], text[n:]


# ---------- 純函式基準（吃參數，可單測）----------
def bpc_random(vocab):
    return math.log2(vocab)


def bpc_gzip(s):
    return len(gzip.compress(s.encode("utf-8"), 9)) * 8 / len(s)


def build_ngram(ids, order):
    s = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    tables = [defaultdict(Counter) for _ in range(order + 1)]
    for i in range(len(s)):
        for m in range(1, order + 1):
            if i - (m - 1) >= 0:
                tables[m][tuple(s[i - (m - 1):i])][s[i]] += 1
    return tables


def bpc_ngram(tables, ids, order, vocab):
    s = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    total_bits = 0.0
    for i in range(len(s)):
        c, p = s[i], None
        for m in range(min(order, i + 1), 0, -1):          # 高階往低階 backoff
            cnt = tables[m].get(tuple(s[i - (m - 1):i]))
            if cnt:
                p = (cnt.get(c, 0) + ADD_K) / (sum(cnt.values()) + ADD_K * vocab)
                break
        if p is None:
            p = 1.0 / vocab
        total_bits += -math.log2(p)
    return total_bits / len(s)


# ---------- 模型：char-level tiny GPT ----------
class Block(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.nh, self.hd = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd))
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(n_embd, 2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        a = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        a = a.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(a, -1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(vocab) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.lnf(self.blocks(x)))


def get_batch():
    ix = torch.randint(len(train_ids) - block_size, (batch_size,))
    x = torch.stack([train_ids[i:i + block_size] for i in ix])
    y = torch.stack([train_ids[i + 1:i + block_size + 1] for i in ix])
    return x, y


def train_gpt():
    m = GPT(V)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(max_iters):
        x, y = get_batch()
        loss = F.cross_entropy(m(x).view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


@torch.no_grad()
def bpc_gpt(m):
    """在整段 val 上逐塊算 per-char NLL（與 n-gram/gzip 同一份 val、公平）。"""
    m.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(val_ids) - 1, block_size):
        chunk = val_ids[i:i + block_size + 1]
        if len(chunk) < 2:
            break
        x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
        tot += F.cross_entropy(m(x).view(-1, V), y.view(-1), reduction="sum").item()
        cnt += y.numel()
    return (tot / cnt) / LN2          # nat→bit


def results():
    load_corpus()
    tables = build_ngram(train_ids, NGRAM_ORDER)
    return [("亂猜 (log2 V)", bpc_random(V)),
            ("gzip -9", bpc_gzip(val_text)),
            (f"{NGRAM_ORDER}-gram (backoff)", bpc_ngram(tables, val_ids, NGRAM_ORDER, V)),
            ("tiny GPT", bpc_gpt(train_gpt()))]


if __name__ == "__main__":
    rows = results()
    print(f"vocab={V}  device=cpu  在同一份 val（{len(val_text)} 字元）上比 BPC\n")
    print(f"{'方法':<22}{'BPC (bits/char)':>16}")
    print("-" * 40)
    for name, b in rows:
        print(f"{name:<22}{b:>16.3f}")
    gpt_b = dict(rows)["tiny GPT"]
    ng_b = [b for nme, b in rows if "gram" in nme][0]
    print(f"\nBPC 越低越好。tiny GPT {gpt_b:.3f} vs {NGRAM_ORDER}-gram {ng_b:.3f}："
          f"{'神經網路贏了笨基準，賺到複雜度 ✅' if gpt_b < ng_b else '沒贏過 n-gram——baseline 很硬，這本身是誠實發現'}")
    print("選的是『下界基準(亂猜/gzip/n-gram)』不是『虛榮基準(GPT-2)』——這才是『選對 baseline』。")
