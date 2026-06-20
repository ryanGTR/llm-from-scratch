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


def build_rope_cache(head_dim: int, max_seq: int, base: float = 10000.0):
    """預先算好每個位置、每個維度對的旋轉角度的 cos/sin。

    第 i 對維度的旋轉頻率 = base^(-2i/head_dim)：低維轉得快、高維轉得慢，
    像時鐘的秒針/分針/時針——不同頻率組合就能唯一編碼位置。
    回傳 cos, sin，形狀 (max_seq, head_dim/2)。
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)          # (max_seq, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """把 q 或 k 依其位置「旋轉」。x: (B, n_head, T, head_dim)。

    把每相鄰兩維 (x1, x2) 當成平面上一個點，旋轉角度 θ：
      x1' = x1·cosθ − x2·sinθ
      x2' = x1·sinθ + x2·cosθ
    旋轉不改變向量長度，只改方向 → q·k 點積會自然變成「相對位置」的函數。
    """
    T = x.shape[2]
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
    x1, x2 = x[..., ::2], x[..., 1::2]        # 偶數維 / 奇數維
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)   # 交錯併回去


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
        # RoPE：預先算好旋轉用的 cos/sin（head_dim 必須是偶數）
        self.use_rope = cfg.use_rope
        if cfg.use_rope:
            cos, sin = build_rope_cache(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, T, C = x.shape  # batch, time(序列長), channels(n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        # 拆成多頭：(B, T, C) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        if self.use_rope:                 # 依位置旋轉 q、k（v 不轉）
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
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


class SwiGLU(nn.Module):
    """Swish-Gated MLP（LLaMA/PaLM 同款）。比普通 MLP 多一條「閘門」。

    普通 MLP：x → Linear → GELU → Linear。
    SwiGLU ：兩條並行投影，一條過 SiLU(swish) 當「閘門」去乘另一條（gating），
             再投影回去：down( silu(gate(x)) * up(x) )。
    閘門讓網路能「動態決定讓多少訊號通過」，表達力更強。
    hidden 取 8/3·n_embd（而非 4·n_embd），讓參數量跟普通 MLP 幾乎相同 → 公平對比。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(round(8 / 3 * cfg.n_embd / 8) * 8)   # ≈8/3·n_embd，湊 8 的倍數
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """一個 Transformer block：attention + MLP，各帶 residual connection。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg) if cfg.use_swiglu else MLP(cfg)

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
        # RoPE 把位置資訊放進 attention 的旋轉裡 → 不需要這個學習式位置 embedding
        self.pos_emb = None if cfg.use_rope else nn.Embedding(cfg.block_size, cfg.n_embd)
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
        x = self.token_emb(idx)
        if self.pos_emb is not None:       # RoPE 模式下位置在 attention 處理，這裡不加
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
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
