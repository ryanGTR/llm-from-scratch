"""A minimal GPT (decoder-only Transformer).

這是整個專案的核心 business logic。結構 = nanoGPT 的精簡版：
  token + position embedding
    -> N x [ LayerNorm -> CausalSelfAttention -> LayerNorm -> MLP ]
    -> LayerNorm -> Linear(vocab)

Java 類比：把它想成一條 filter chain，每個 Block 都對隱藏狀態做一次
「看上下文 -> 更新自己的理解」。最後一層輸出每個 token 的下一字機率。
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig


class RMSNorm(nn.Module):
    """Root Mean Square Norm（LLaMA/Mistral 同款）。比 LayerNorm 更簡單。

    LayerNorm：(x - 均值) / 標準差，再縮放平移 —— 要算均值跟變異數。
    RMSNorm　：x / 均方根(RMS)，再縮放 —— 不減均值、不要 bias，更省、更穩，
    實務上效果跟 LayerNorm 一樣好。現代 LLM 幾乎都換成它。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def make_norm(cfg: GPTConfig):
    """依設定回傳 RMSNorm 或 LayerNorm，讓 model 可一鍵切換。"""
    if cfg.use_rmsnorm:
        return RMSNorm(cfg.n_embd)
    return nn.LayerNorm(cfg.n_embd, bias=cfg.bias)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask（只能看左邊，不能偷看未來）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd 必須能被 n_head 整除"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        # 一次算出 query / key / value（三份各 n_embd 維）
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # 下三角矩陣：第 i 個 token 只能 attend 到 <= i 的位置
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, time(序列長), channels(n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        # 拆成多頭：(B, T, C) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # attention scores，scale 防止數值爆掉
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                       # 加權平均出新的 representation
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 多頭併回去
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward：每個位置各自過一個小 MLP。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """一個 Transformer block：attention + MLP，各帶 residual connection。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # residual：x = x + f(x)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.vocab_size > 0, "vocab_size 還沒設定，先跑 prepare_data"
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying：輸入 embedding 和輸出層共用權重（省參數、常見技巧）
        self.token_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "輸入超過 block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)             # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross entropy：把 (B,T,vocab) 攤平成 (B*T, vocab) 對 (B*T,) 算
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回歸生成：吐一個 token、接回輸入、再吐下一個。"""
        for _ in range(max_new_tokens):
            # context 超過 block_size 就裁掉最舊的部分
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # 只看最後一個位置
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
