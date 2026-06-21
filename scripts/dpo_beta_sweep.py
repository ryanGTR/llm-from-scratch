"""DPO 精修：掃 β（KL 約束旋鈕），看「對齊力度」的三個取捨面。

β 是 DPO 唯一的核心旋鈕。常見直覺「β 大＝KL 罰得重＝緊貼 reference」——但這份實測在
「固定訓練步數」下**翻了這個直覺**：DPO loss = −logσ(β·margin)，飽和點在 margin≈1/β。
  β 小 → 飽和門檻 1/β 很大 → 優化器把 logπ gap 一路推到超大 → **漂移大、過度優化**
  β 大 → 小 gap 就飽和 → policy 幾乎不用動 → **漂移小、對齊溫和**（偏好學得淺一點）
（無限步數的漸近行為另當別論；這裡量的是真實情境：固定算力預算下 β 怎麼影響結果。）

用 format 軸（連貫 vs 退化重複，模型學得動）掃 β，每個 β 量三件事：
  1. held-out 偏好準確率：學到偏好了嗎（logπ 排名）
  2. 對 reference 的漂移（每 token |Δlogπ|，KL 代理）：離 SFT 多遠
  3. **實際生成的重複率**：對齊有沒有「真的改變產出」——這才是行為層的兌現，
     不是只在 logπ 排名上贏（延續本專案鐵則：量行為、不只量代理指標）。

  python scripts/dpo_beta_sweep.py
"""

import importlib
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig          # noqa: E402
from src.model import GPT                  # noqa: E402
from src.tokenizer import load_tokenizer   # noqa: E402

dpo = importlib.import_module("pipeline.06_dpo")
ART = ROOT / "artifacts"

BETAS = [0.02, 0.1, 0.5]
ITERS = 300
BATCH = 16
LR = 1e-4
BLOCK = 128
SEED = 1337


def load_sft(device):
    ck = torch.load(ART / "sft_ckpt.pt", map_location=device)
    m = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    m.load_state_dict(ck["model"])
    return m, ck


@torch.no_grad()
def drift_vs_ref(policy, ref, tok, rows, device):
    """每 token 平均 |logπ_policy − logπ_ref|（在 held-out chosen+rejected 上）= KL 代理。"""
    tot = 0.0
    n = 0
    for r in rows:
        for key in ("chosen", "rejected"):
            ids, m = dpo.build_example(tok, r["prompt"], r[key], BLOCK)
            X = torch.tensor([ids], device=device)
            M = torch.tensor([m], dtype=torch.float, device=device)
            ntok = max(1.0, M[:, 1:].sum().item())
            lp_p = dpo.seq_logp(policy, X, M).item() / ntok
            lp_r = dpo.seq_logp(ref, X, M).item() / ntok
            tot += abs(lp_p - lp_r)
            n += 1
    return tot / n


@torch.no_grad()
def gen_repetition(model, tok, prompts, device):
    """生成續寫的重複率＝1 − distinct-2（重複 bigram 比例）。高=陷入重複迴圈。"""
    torch.manual_seed(SEED)        # 固定取樣，β 之間可比
    rates = []
    for q in prompts:
        ids = torch.tensor([tok.encode(dpo.PROMPT_TMPL.format(q=q))], device=device)
        out = model.generate(ids, 40, temperature=0.8, top_p=0.9)[0].tolist()
        gen = out[ids.shape[1]:]
        bg = list(zip(gen, gen[1:]))
        if bg:
            rates.append(1 - len(set(bg)) / len(bg))
    return sum(rates) / len(rates)


def train_dpo(ref, ck, rows, beta, device):
    """從 SFT 複製一顆 policy，用給定 β 訓 ITERS 步，回傳訓好的 policy。"""
    policy = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    policy.load_state_dict(ref.state_dict())
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(ITERS):
        idx = torch.randint(len(rows), (BATCH,), generator=g).tolist()
        X, M = dpo.collate([rows[i] for i in idx], tok, BLOCK, device)
        plp = dpo.seq_logp(policy, X, M)
        with torch.no_grad():
            rlp = dpo.seq_logp(ref, X, M)
        loss, _, _ = dpo.dpo_loss(plp, rlp, beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return policy.eval()


def make_plot(betas, accs, drifts, reps, sft_acc, sft_rep):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = range(len(betas))
    labels = [str(b) for b in betas]

    # 左：偏好準確率 + 生成重複率
    ax1.plot(x, [a * 100 for a in accs], "-o", color="#2ca02c", label="held-out 偏好準確率")
    ax1.axhline(sft_acc * 100, ls=":", color="#2ca02c", alpha=0.5, label="SFT 基準（偏好）")
    ax1b = ax1.twinx()
    ax1b.plot(x, [r * 100 for r in reps], "-s", color="#d62728", label="生成重複率")
    ax1b.axhline(sft_rep * 100, ls=":", color="#d62728", alpha=0.5, label="SFT 基準（重複）")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels)
    ax1.set_xlabel("β（KL 約束強度）"); ax1.set_ylabel("偏好準確率 (%)", color="#2ca02c")
    ax1b.set_ylabel("生成重複率 (%)", color="#d62728")
    ax1.set_title("學到偏好了嗎 + 產出真的改善了嗎")
    ax1.legend(loc="center left", fontsize=8); ax1b.legend(loc="center right", fontsize=8)
    ax1.grid(alpha=0.2)

    # 右：對 reference 的漂移
    ax2.plot(x, drifts, "-^", color="#1f77b4")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels)
    ax2.set_xlabel("β（KL 約束強度）"); ax2.set_ylabel("每 token |Δlogπ|（離 SFT 多遠）")
    # 反直覺：β 越「小」漂移越「大」。margin 目標≈1/β，小 β 追求超大 gap → 過度優化。
    ax2.set_title("反直覺：β 越小、漂移越大（margin 目標≈1/β）")
    ax2.grid(alpha=0.2)

    out = ART / "dpo_beta_sweep.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"\n圖 → {out}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(ART / "tokenizer.json")
    rows = [__import__("json").loads(l) for l in
            (ART / "dpo_format.jsonl").read_text(encoding="utf-8").splitlines()]
    held = [__import__("json").loads(l) for l in
            (ART / "dpo_format_heldout.jsonl").read_text(encoding="utf-8").splitlines()]
    prompts = [r["prompt"] for r in held[:20]]

    ref, ck = load_sft(device)
    ref.eval()
    sft_acc = dpo.heldout_pref_acc(ref, tok, held, BLOCK, device)
    sft_rep = gen_repetition(ref, tok, prompts, device)
    print(f"SFT 基準：偏好 {sft_acc*100:.0f}%、生成重複率 {sft_rep*100:.0f}%\n")
    print(f"{'β':>6} | {'偏好準確率':>8} | {'漂移|Δlogπ|':>10} | {'生成重複率':>8}")
    print("-" * 46)

    accs, drifts, reps = [], [], []
    for beta in BETAS:
        pol = train_dpo(ref, ck, rows, beta, device)
        acc = dpo.heldout_pref_acc(pol, tok, held, BLOCK, device)
        dr = drift_vs_ref(pol, ref, tok, held, device)
        rep = gen_repetition(pol, tok, prompts, device)
        accs.append(acc); drifts.append(dr); reps.append(rep)
        print(f"{beta:>6} | {acc*100:>7.0f}% | {dr:>10.3f} | {rep*100:>7.0f}%")

    try:
        make_plot(BETAS, accs, drifts, reps, sft_acc, sft_rep)
    except Exception as e:
        print(f"（略過畫圖：{e}）")
