"""tiny_observability.py — 在 CPU 上親手看「可觀測性」：冷啟動、延遲分布、metrics。

書本 Ch6（服務化與可觀測）的「💻 在你的機器上」配套程式。把一顆小模型包成一個
最小「推論服務」，量它的延遲——示範可觀測性的價值：**你不量就不知道、量了立刻看到**。

它做三件事：
  1) **冷啟動**：第一個請求通常比之後慢（lazy 初始化、執行緒池暖機、配置快取）。量給你看。
  2) **延遲分布**：服務不能只看平均——要看 p50/p95（尾延遲才是使用者體感）。
  3) **metrics 計數**：每個請求自動計數、累計 token——一裝上就有數據可看。

模型用隨機初始化即可——延遲跟模型好不好無關，只跟「算多少」有關。

用法：
    python tiny_observability.py     # 純 CPU、十幾秒、不需語料、不需 GPU
"""

import time
import torch
import torch.nn as nn
from torch.nn import functional as F

block_size, n_embd, n_head, n_layer, vocab_size = 128, 128, 4, 4, 96
torch.manual_seed(0)
torch.set_num_threads(4)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
        self.register_buffer("mask", torch.triu(
            torch.full((block_size, block_size), float("-inf")), diagonal=1))

    def forward(self, x):
        T = x.shape[1]
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T], need_weights=False)
        return x + a + self.mlp(self.ln2(x + a))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits = self.head(self.lnf(self.blocks(
                self.tok(idx[:, -block_size:]) + self.pos(
                    torch.arange(min(idx.shape[1], block_size))))))
            idx = torch.cat([idx, logits[:, -1, :].argmax(-1, keepdim=True)], 1)
        return idx


class Service:
    """最小推論服務：常駐模型 + 每個請求自動記 metrics（計數、累計 token、延遲）。"""

    def __init__(self):
        self.model = GPT().eval()
        self.n_requests = 0
        self.n_tokens = 0
        self.latencies = []

    def generate(self, prompt, n_new=40):
        t0 = time.perf_counter()
        out = self.model.generate(prompt, n_new)
        dt = (time.perf_counter() - t0) * 1000          # ms
        self.n_requests += 1
        self.n_tokens += n_new
        self.latencies.append(dt)
        return out, dt


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


if __name__ == "__main__":
    svc = Service()
    prompt = torch.randint(0, vocab_size, (1, 8))
    print(f"模型 {sum(p.numel() for p in svc.model.parameters())/1e6:.2f}M，device=cpu\n")

    print("=== 打 30 個請求，看延遲 ===")
    for _ in range(30):
        svc.generate(prompt)
    cold = svc.latencies[0]
    warm = svc.latencies[1:]
    print(f"  冷啟動（第 1 個請求）：{cold:6.1f} ms")
    print(f"  暖機後 p50：          {pct(warm, 50):6.1f} ms")
    print(f"  暖機後 p95：          {pct(warm, 95):6.1f} ms")
    print(f"  冷/暖倍數：           {cold / pct(warm, 50):6.1f}×")

    print("\n=== /metrics（一裝上就有數據）===")
    print(f"  requests_total      = {svc.n_requests}")
    print(f"  tokens_generated    = {svc.n_tokens}")
    print(f"  latency_p50_ms      = {pct(warm, 50):.1f}")
    print(f"  latency_p95_ms      = {pct(warm, 95):.1f}")
    print("\n→ 『好像第一次比較慢』變成『冷 X ms vs 暖 Y ms』的事實。"
          "服務不能只看平均，尾延遲(p95)才是體感。")
