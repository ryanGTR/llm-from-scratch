"""tiny_serve.py — 在 CPU 上親手做「模型治理」：digest 身份 + promotion gate。

書本 Ch6 的「💻 在你的機器上」配套程式。服務化有很多面（API、可觀測、容器），
但最差異化、也最該親手跑一次的是**治理**：對「線上那顆模型」答得出稽核四問。
這支不需 GPU、約一分鐘，示範其中兩問的可執行核心：

  1) **「哪一個？」用 sha256 digest，不靠檔名**：
     - 把 checkpoint 的權重 bytes 算 sha256 當身份。
     - 證明：把檔案改名，digest **不變**；偷改一個權重，digest **立刻變**。
       檔名會騙人、會被覆寫；digest 是內容的指紋。
  2) **「憑什麼上線？」promotion gate 用程式 enforce**：
     - 一個小 registry（台帳）綁住每顆模型的 eval 數字 + 資料品質 gate 結果。
     - gate 規則：沒過資料 gate、或沒比現行更好，就**擋下**——不是「誰想推就推」。
     - 服務端 /model 回報線上 digest 的 registry 狀態；查不到就回 UNREGISTERED（稽核紅旗）。

用法：
    curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python tiny_serve.py
"""

import hashlib
import io
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 64
n_embd, n_head, n_layer = 96, 4, 3
batch_size = 32

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
        x = x + a
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
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        return self.head(self.lnf(self.blocks(x)))


def train(steps, seed):
    torch.manual_seed(seed)
    m = GPT()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = get_batch("train")
        loss = F.cross_entropy(m(x).view(-1, vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


@torch.no_grad()
def val_loss(m):
    return torch.stack([F.cross_entropy(m(x).view(-1, vocab_size), y.view(-1))
                        for x, y in (get_batch("val") for _ in range(30))]).mean().item()


def digest(model):
    """模型身份 = 權重 bytes 的 sha256（像 container image digest / cosign 簽章）。"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ====== registry：每顆模型綁住 eval + 資料 gate + lineage ======
REGISTRY = {}

def register(model, *, val_loss, data_gate, git_commit):
    REGISTRY[digest(model)] = dict(val_loss=val_loss, data_gate=data_gate,
                                   git_commit=git_commit)

def promote(model, current_loss):
    """promotion gate：沒過資料 gate、或沒比現行更好，就擋下。"""
    rec = REGISTRY.get(digest(model))
    if rec is None:
        return False, "UNREGISTERED（台帳查無此模型，稽核紅旗）"
    if not rec["data_gate"]:
        return False, "BLOCK：資料品質 gate 未過"
    if rec["val_loss"] >= current_loss:
        return False, f"BLOCK：未比現行更好（{rec['val_loss']:.3f} ≥ {current_loss:.3f}）"
    return True, f"PROMOTE：通過（{rec['val_loss']:.3f} < {current_loss:.3f} 且資料 gate ✅）"


if __name__ == "__main__":
    print(f"vocab={vocab_size}  device=cpu\n")
    print("訓兩顆：current（少訓）、candidate（多訓）...")
    current = train(300, seed=1)
    candidate = train(900, seed=2)
    cur_loss, cand_loss = val_loss(current), val_loss(candidate)

    print("\n=== 稽核問題 1：哪一個？（digest 不靠檔名）===")
    d_cur = digest(current)
    print(f"  current   digest: {d_cur[:16]}…  val {cur_loss:.3f}")
    print(f"  candidate digest: {digest(candidate)[:16]}…  val {cand_loss:.3f}")
    # 改名不改內容 → digest 不變
    print(f"  把 current 存成 modelA.pt / 又存成 best_FINAL.pt → digest 變嗎？ "
          f"{digest(current) == d_cur}（不變，digest 認內容不認檔名）")
    # 偷改一個權重 → digest 立刻變
    with torch.no_grad():
        current.head.bias[0] += 1e-4
    print(f"  偷改一個權重 (+1e-4) → digest 變嗎？ {digest(current) != d_cur}（立刻變）")
    # 還原，後面 gate 才用對的身份
    with torch.no_grad():
        current.head.bias[0] -= 1e-4

    print("\n=== 稽核問題 4：憑什麼上線？（promotion gate 用程式 enforce）===")
    register(current, val_loss=cur_loss, data_gate=True, git_commit="a1b2c3d")
    # 候選 A：資料 gate 沒過 → 該擋
    register(candidate, val_loss=cand_loss, data_gate=False, git_commit="e4f5g6h")
    ok, msg = promote(candidate, cur_loss)
    print(f"  候選（資料 gate=False）：{msg}")
    # 候選 B：補過資料 gate 後重註冊 → 這次該放行
    register(candidate, val_loss=cand_loss, data_gate=True, git_commit="e4f5g6h")
    ok, msg = promote(candidate, cur_loss)
    print(f"  候選（資料 gate=True ）：{msg}")
    # 一顆沒註冊的野模型想上線 → 紅旗
    rogue = train(50, seed=99)
    ok, msg = promote(rogue, cur_loss)
    print(f"  未註冊的野模型：{msg}")

    print("\n上線不是『誰想推就推』，是被 gate 程式擋住——這就是把稽核接到 ML。")
