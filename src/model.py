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


def apply_rope(x, cos, sin, offset=0):
    """把 q 或 k 依其位置「旋轉」。x: (B, n_head, T, head_dim)。

    把每相鄰兩維 (x1, x2) 當成平面上一個點，旋轉角度 θ：
      x1' = x1·cosθ − x2·sinθ
      x2' = x1·sinθ + x2·cosθ
    旋轉不改變向量長度，只改方向 → q·k 點積會自然變成「相對位置」的函數。
    offset：KV-cache 增量生成時，新 token 的絕對位置不是從 0 起，要用 offset 對齊。
    """
    T = x.shape[2]
    cos = cos[offset:offset + T].view(1, 1, T, -1)
    sin = sin[offset:offset + T].view(1, 1, T, -1)
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
        self.head_dim = cfg.n_embd // cfg.n_head
        self.use_flash = cfg.use_flash
        self.dropout_p = cfg.dropout
        # GQA：n_kv_head 組 key/value 給 n_head 個 query 共用（0=標準 MHA）
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        assert cfg.n_head % self.n_kv_head == 0, "n_head 要能被 n_kv_head 整除"
        self.kv_dim = self.n_kv_head * self.head_dim
        # q 是 n_embd 維，k/v 各只有 kv_dim 維（GQA 時更小 → 省 KV-cache）
        self.c_attn = nn.Linear(cfg.n_embd, cfg.n_embd + 2 * self.kv_dim, bias=cfg.bias)
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
        self.cache_k = self.cache_v = None   # KV-cache：增量生成時存過去的 k/v

    def forward(self, x, pos_offset=0, use_cache=False):
        B, T, C = x.shape  # batch, time(序列長), channels(n_embd)
        hd = self.head_dim
        # q 有 n_head 個頭、k/v 只有 n_kv_head 個頭
        q, k, v = self.c_attn(x).split([self.n_embd, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, hd).transpose(1, 2)
        if self.use_rope:                 # 依位置旋轉 q、k（v 不轉），offset 給快取對位
            q = apply_rope(q, self.rope_cos, self.rope_sin, pos_offset)
            k = apply_rope(k, self.rope_cos, self.rope_sin, pos_offset)
        if use_cache:                     # KV-cache：把新 k/v 接到快取後面（存 GQA 前的小份）
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=2)
                v = torch.cat([self.cache_v, v], dim=2)
            self.cache_k, self.cache_v = k, v
        if self.n_kv_head != self.n_head:  # GQA：把每組 k/v 複製給該組的 query 頭共用
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if use_cache:
            # 快取路徑：query 在絕對位置 [pos_offset, pos_offset+T)，key 在 [0, T_kv)
            # 顯式因果遮罩（一般化版本，prefill 與單 token 增量都對）
            T_kv = k.shape[2]
            qpos = torch.arange(pos_offset, pos_offset + T, device=x.device)
            allow = (torch.arange(T_kv, device=x.device)[None, :] <= qpos[:, None])
            allow = allow.view(1, 1, T, T_kv)
            if self.use_flash:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
            else:
                att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
                att = att.masked_fill(~allow, float("-inf"))
                y = F.softmax(att, dim=-1) @ v
        elif self.use_flash:
            # FlashAttention：torch 內建、不攤開 T×T 矩陣 → 記憶體 O(T²)→O(T)。
            # 結果跟下面樸素版「數學上完全一樣」，只是算法更省。is_causal 幫忙做因果遮罩。
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0)
        else:
            # 樸素版（教學用，把整個 T×T attention 矩陣攤出來）
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                   # 加權平均出新的 representation
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

    def forward(self, x, pos_offset=0, use_cache=False):
        x = x + self.attn(self.ln_1(x), pos_offset, use_cache)   # residual：x = x + f(x)
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

    def reset_cache(self):
        """清空所有層的 KV-cache（每次重新生成前呼叫）。"""
        for block in self.blocks:
            block.attn.cache_k = block.attn.cache_v = None

    def forward(self, idx, targets=None, pos_offset=0, use_cache=False):
        B, T = idx.shape
        assert pos_offset + T <= self.cfg.block_size, "位置超過 block_size"
        x = self.token_emb(idx)
        if self.pos_emb is not None:       # RoPE 模式下位置在 attention 處理，這裡不加
            x = x + self.pos_emb(torch.arange(pos_offset, pos_offset + T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, pos_offset, use_cache)
        x = self.ln_f(x)
        logits = self.head(x)             # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross entropy：把 (B,T,vocab) 攤平成 (B*T, vocab) 對 (B*T,) 算
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def _sample(self, logits, temperature, top_k, top_p, min_p):
        """從最後一個位置的 logits 砍候選後抽一個 token。"""
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        if top_p is not None:                      # nucleus：累積機率截斷
            s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()   # 保留剛跨過門檻的那個
            remove[..., 0] = False
            logits[remove.scatter(1, s_idx, remove)] = float("-inf")
        if min_p is not None:                      # 相對於峰值的門檻
            probs = F.softmax(logits, dim=-1)
            logits[probs < min_p * probs.max(dim=-1, keepdim=True).values] = float("-inf")
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=None, min_p=None, use_kv_cache=False):
        """自回歸生成：吐一個 token、接回輸入、再吐下一個。

        砍候選法：top_k=固定數量；top_p=nucleus 累積；min_p=相對峰值（2024）。
        use_kv_cache=True：快取過去 token 的 k/v，每步只算「新 token」→ O(T²)→O(T)，
        生成更快（結果與不快取「完全一樣」，只是算法更省）。
        """
        if use_kv_cache:
            self.reset_cache()
            logits, _ = self(idx, use_cache=True)          # prefill 整段 prompt
            pos = idx.shape[1]
            for _ in range(max_new_tokens):
                next_id = self._sample(logits, temperature, top_k, top_p, min_p)
                idx = torch.cat((idx, next_id), dim=1)
                logits, _ = self(next_id, pos_offset=pos, use_cache=True)  # 只餵新 token
                pos += 1
            self.reset_cache()
            return idx

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]   # context 超過 block_size 裁掉最舊
            logits, _ = self(idx_cond)
            next_id = self._sample(logits, temperature, top_k, top_p, min_p)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
