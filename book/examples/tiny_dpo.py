"""tiny_dpo.py — 在 CPU 上親手看 DPO：margin 怎麼動、train-acc 怎麼騙你。

書本 Ch7 的「💻 在你的機器上」配套程式。流程：
  1) 用 Ch1 的 char-level 小 GPT 在莎士比亞上預訓練一個 base（當 reference）。
  2) 複製一份當 policy，用 DPO 損失（封閉式、免 reward model、免 RL）做偏好優化。
  3) 設「兩種偏好軸」對照——一種模型學得動、一種學不動——印出 train-acc 與 held-out-acc，
     親眼看到課本鐵則：**train-acc 都衝 100%，held-out 才說真話。**

兩種偏好軸：
  A. 可學（real vs scrambled）：chosen = 真實片段、rejected = 同片段打亂字元順序。
     「真的比亂的好」是通則 → train 與 held-out 都該高。
  B. 學不動（隨機標籤）：chosen / rejected 都是真實片段，誰是 chosen 用擲硬幣亂指。
     規則無語義、無法類推 → 只能把 train 背起來（acc→100%），held-out 停在 ~50% 擲硬幣。

用法：
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python tiny_dpo.py            # 純 CPU，約 2–3 分鐘

DPO 損失本身只有 4 行（見 dpo_loss），是本章「逐行走讀」的主角。
"""

import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

_SMOKE = bool(os.environ.get("BOOK_SMOKE"))   # CI 煙霧測試：極少步數只驗「跑得動」

# ---- 超參數（小到 CPU 也跑得動）----
block_size = 64
n_embd     = 128
n_head     = 4
n_layer    = 3
pre_iters  = 30 if _SMOKE else 1500          # base 預訓練步數
dpo_iters  = 20 if _SMOKE else 300           # DPO 步數
batch_size = 32
n_pairs    = 32 if _SMOKE else 256           # 每軸偏好對：一半 train、一半 held-out
beta       = 0.1           # DPO 溫度
torch.manual_seed(1337)

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


# ---- 模型：與 Ch1 的 tiny_gpt 同構（樸素 attention，看得懂優先）----
class Head(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.key   = nn.Linear(n_embd, hd, bias=False)
        self.query = nn.Linear(n_embd, hd, bias=False)
        self.value = nn.Linear(n_embd, hd, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        att = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(att, dim=-1) @ v


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        hd = n_embd // n_head
        self.heads = nn.ModuleList([Head(hd) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHead()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T))
        return self.head(self.lnf(self.blocks(x)))           # (B,T,vocab)


def seq_logprob(model, seq):
    """整段序列的對數機率：Σ_t log p(token_t | token_<t)。回傳 (B,)。"""
    logits = model(seq[:, :-1])                              # (B,T-1,vocab)
    logp = F.log_softmax(logits, dim=-1)
    tgt = seq[:, 1:].unsqueeze(-1)                           # (B,T-1,1)
    return logp.gather(-1, tgt).squeeze(-1).sum(dim=1)       # (B,)


# ============ DPO 損失：本章逐行走讀的主角 ============
def dpo_loss(model, ref, chosen, rejected, beta):
    pol_ch,  pol_rej  = seq_logprob(model, chosen), seq_logprob(model, rejected)
    ref_ch,  ref_rej  = seq_logprob(ref,   chosen), seq_logprob(ref,   rejected)
    margin = (pol_ch - ref_ch) - (pol_rej - ref_rej)        # chosen 比 rejected 多漲多少
    loss = -F.logsigmoid(beta * margin).mean()              # 把 margin 推大
    acc = (margin > 0).float().mean()                       # margin>0 = 偏好排對了
    return loss, acc.item()
# =====================================================


def pretrain():
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for it in range(pre_iters):
        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f"  base 預訓練完成（{pre_iters} 步，val loss {est_val(model):.3f}）")
    return model


@torch.no_grad()
def est_val(model):
    return torch.stack([
        F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        for x, y in (get_batch("val") for _ in range(20))]).mean().item()


def real_segments(k):
    """從 val 區抓 k 段長度 block_size 的真實片段。"""
    ix = torch.randint(len(val_data) - block_size, (k,))
    return torch.stack([val_data[i:i + block_size] for i in ix])


def scramble(seg):
    """打亂每段內字元順序（破壞結構，但字元組成不變）。"""
    out = seg.clone()
    for b in range(out.shape[0]):
        out[b] = out[b][torch.randperm(block_size)]
    return out


def build_axis(kind):
    """回傳 (chosen, rejected)，各 n_pairs 段。"""
    if kind == "learnable":                  # 真實 vs 打亂
        chosen = real_segments(n_pairs)
        rejected = scramble(chosen)
    else:                                    # 兩段都真實，誰是 chosen 用擲硬幣
        a, b = real_segments(n_pairs), real_segments(n_pairs)
        flip = torch.rand(n_pairs) < 0.5
        chosen = torch.where(flip.unsqueeze(1), a, b)
        rejected = torch.where(flip.unsqueeze(1), b, a)
    return chosen, rejected


@torch.no_grad()
def eval_acc(model, ref, chosen, rejected):
    pol = seq_logprob(model, chosen) - seq_logprob(model, rejected)
    ref_m = seq_logprob(ref, chosen) - seq_logprob(ref, rejected)
    return ((pol - ref_m) > 0).float().mean().item()


def run_dpo(base, kind):
    chosen, rejected = build_axis(kind)
    half = n_pairs // 2
    tr_c, tr_r = chosen[:half], rejected[:half]              # train
    ho_c, ho_r = chosen[half:], rejected[half:]              # held-out
    ref = base                                               # 凍結的 reference
    model = TinyGPT(); model.load_state_dict(base.state_dict())   # policy = base 的副本
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for it in range(dpo_iters):
        ix = torch.randint(half, (batch_size,))
        loss, _ = dpo_loss(model, ref, tr_c[ix], tr_r[ix], beta)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    tr_acc = eval_acc(model, ref, tr_c, tr_r)
    ho_acc = eval_acc(model, ref, ho_c, ho_r)
    return tr_acc, ho_acc


if __name__ == "__main__":
    print(f"vocab={vocab_size}  device=cpu")
    base = pretrain()
    print(f"\n{'偏好軸':<32}{'train-acc':>11}{'held-out-acc':>14}")
    print("-" * 57)
    for kind, label in (("learnable", "A 可學（真實 vs 打亂）"),
                        ("random",    "B 學不動（隨機標籤）")):
        tr, ho = run_dpo(base, kind)
        print(f"{label:<32}{tr*100:>10.1f}%{ho*100:>13.1f}%")
    print("\n看 held-out：A 兩邊都高＝真的學會「真>亂」這個通則；"
          "\nB train 背到高分、held-out 掉回擲硬幣＝只是死背。train-acc 會騙你。")
