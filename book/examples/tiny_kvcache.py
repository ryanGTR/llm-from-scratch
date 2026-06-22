"""tiny_kvcache.py — 在 CPU 上親手量 KV-cache：先證它沒算錯，再看它快多少。

書本 Ch3 的「💻 在你的機器上」配套程式。自回歸生成有一個 $O(T^2)$ 的浪費：
每吐一個 token，樸素做法都把前面**所有** token 的 key/value 重算一遍。KV-cache 把算過的
key/value 快取起來，每步只算「新 token」那一個，理論上 $O(T^2)\to O(T)$。

這支做兩件事：
  1) **先證對**：對同一顆模型、同一個起點、greedy 解碼，cached 與 naive 必須吐出
     **逐 token 完全相同**的序列——KV-cache 是「省」，不是「近似」，數學上一字不差。
  2) **再看快**：量兩者生成 500 個 token 的牆鐘時間，看 CPU 上省多少。

模型用隨機初始化即可——**計時跟模型好不好無關**，只跟「算多少」有關。

用法：
    python tiny_kvcache.py        # 純 CPU，約十幾秒，不需語料、不需 GPU
"""

import math
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

_SMOKE = bool(os.environ.get("BOOK_SMOKE"))   # CI 煙霧測試：少生成幾個只驗「跑得動」

block_size = 1024       # 最大 context（夠大，讓章末習題的 n_new=1000 也不爆）
n_embd     = 128
n_head     = 4
n_layer    = 4
vocab_size = 96
n_new      = 60 if _SMOKE else 500        # 生成多少 token
torch.manual_seed(0)
head_dim = n_embd // n_head


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def _split(self, t, B, T):
        return t.view(B, T, n_head, head_dim).transpose(1, 2)   # (B,nh,T,hd)

    def forward(self, x):
        """樸素：對整段 x 算 attention（含因果遮罩）。"""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q, k, v = self._split(q, B, T), self._split(k, B, T), self._split(v, B, T)
        att = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

    def step(self, x_t, cache):
        """KV-cache：x_t 只有「新的那一個 token」(B,1,C)；把它的 k/v 接到 cache 後面。"""
        B = x_t.shape[0]
        q, k, v = self.qkv(x_t).split(n_embd, dim=2)
        q, k, v = self._split(q, B, 1), self._split(k, B, 1), self._split(v, B, 1)
        if cache is None:
            k_all, v_all = k, v
        else:
            k_all = torch.cat([cache[0], k], dim=2)            # 接上歷史 key
            v_all = torch.cat([cache[1], v], dim=2)            # 接上歷史 value
        att = q @ k_all.transpose(-2, -1) / math.sqrt(head_dim)  # 不需遮罩：cache 全是過去
        y = F.softmax(att, dim=-1) @ v_all
        out = self.proj(y.transpose(1, 2).contiguous().view(B, 1, n_embd))
        return out, (k_all, v_all)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = Attn()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def step(self, x, cache):
        a, new_cache = self.attn.step(self.ln1(x), cache)
        x = x + a
        return x + self.mlp(self.ln2(x)), new_cache


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    @torch.no_grad()
    def generate_naive(self, idx, n):
        """每步把整段（越來越長）重跑一次——$O(T^2)$ 的重複勞動。"""
        for _ in range(n):
            logits = self(idx[:, -block_size:])
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)    # greedy，確保可對拍
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    @torch.no_grad()
    def generate_cached(self, idx, n):
        """先把 prompt 灌進 cache，之後每步只餵「新 token」一個——$O(T)$。"""
        caches = [None] * n_layer
        pos = 0
        # 灌入 prompt：逐 token 建好 cache
        for t in range(idx.shape[1]):
            x = self.tok(idx[:, t:t + 1]) + self.pos(torch.arange(pos, pos + 1))
            for i, b in enumerate(self.blocks):
                x, caches[i] = b.step(x, caches[i])
            pos += 1
        logits = self.head(self.lnf(x))
        out = idx
        for _ in range(n):
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
            x = self.tok(nxt) + self.pos(torch.arange(pos, pos + 1))
            for i, b in enumerate(self.blocks):
                x, caches[i] = b.step(x, caches[i])
            pos += 1
            logits = self.head(self.lnf(x))
        return out


if __name__ == "__main__":
    model = GPT().eval()
    print(f"參數 {sum(p.numel() for p in model.parameters())/1e6:.2f}M，"
          f"context 上限 {block_size}，生成 {n_new} token，device=cpu\n")
    prompt = torch.randint(0, vocab_size, (1, 8))

    t0 = time.perf_counter(); a = model.generate_naive(prompt, n_new)
    t_naive = time.perf_counter() - t0
    t0 = time.perf_counter(); b = model.generate_cached(prompt, n_new)
    t_cached = time.perf_counter() - t0

    same = torch.equal(a, b)
    print(f"逐 token 完全相同？ {same}（KV-cache 是省、不是近似）")
    print(f"naive  （每步重算全序列）：{t_naive:6.2f} s")
    print(f"cached （每步只算新 token）：{t_cached:6.2f} s")
    print(f"加速：{t_naive / t_cached:.2f}×")
