"""Stage 6 — DPO（Direct Preference Optimization）：後訓練的偏好對齊。

SFT 教模型「會聽話的格式」；DPO 再教它「在兩個回答之間，偏好較好的那個」。
不需要訓練一個 reward model（那是 RLHF/PPO 的做法），DPO 用一條封閉式損失直接優化：

  loss = -log σ( β·[ (logπ(chosen) - logπ_ref(chosen)) - (logπ(rejected) - logπ_ref(rejected)) ] )

直覺：拉高「policy 相對 reference 對 chosen 的對數機率」、壓低 rejected 的；β 控制
偏離 reference 的力度（小=更貼 reference、大=更敢動）。reference 是凍結的 SFT 模型，
當「錨」防止 policy 為了迎合偏好而崩壞（reward hacking / 退化）。

Java 類比：reference 像「上一版穩定 build」，DPO 只允許在它附近做被偏好資料背書的微調。

  python pipeline/06_dpo.py --iters 1500 --beta 0.1
"""

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig          # noqa: E402
from src.model import GPT                  # noqa: E402
from src.tokenizer import load_tokenizer   # noqa: E402

ART = Path("artifacts")
# 跟 SFT 同款對話格式，模型才認得（05_sft.py 的 TEMPLATE）
PROMPT_TMPL = "問：{q}\n答："
RESP_SUFFIX = "\n\n"


def build_example(tok, prompt, response, block_size):
    """把一筆 (prompt, response) 編成 token 序列 + 「回答段遮罩」。

    回傳 (ids, mask)：mask[i]=1 代表第 i 個 token 屬於回答段（DPO 只在回答段算 logπ，
    指令段不算——我們在乎的是「模型怎麼回答」，不是「會不會背題目」）。
    """
    p_ids = tok.encode(PROMPT_TMPL.format(q=prompt))
    r_ids = tok.encode(response + RESP_SUFFIX)
    ids = (p_ids + r_ids)[:block_size]
    mask = ([0] * len(p_ids) + [1] * len(r_ids))[:block_size]
    return ids, mask


def collate(rows, tok, block_size, device):
    """把一個 batch 的偏好對打包成右側 padding 的張量。

    causal LM + 右側 padding 是安全的：真實 token 只 attend 左邊，永遠看不到尾巴的
    pad；而 pad 位置的 logits 我們用 mask 排除在 loss 之外，不影響結果。
    """
    seqs, masks = [], []
    for r in rows:
        for key in ("chosen", "rejected"):
            ids, m = build_example(tok, r["prompt"], r[key], block_size)
            seqs.append(ids)
            masks.append(m)
    maxlen = max(len(s) for s in seqs)
    X = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    M = torch.zeros(len(seqs), maxlen, dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        X[i, :len(s)] = torch.tensor(s)
        M[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    X, M = X.to(device), M.to(device)
    # chosen 在偶數列、rejected 在奇數列（collate 交錯放）
    return X, M


def seq_logp(model, X, M):
    """每條序列「回答段」的對數機率總和 = Σ log π(token)。

    x = X[:, :-1] 預測 y = X[:, 1:]；遮罩取 M[:, 1:] 對齊到「被預測的那個 token」。
    """
    logits, _ = model(X[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (tok_logp * M[:, 1:]).sum(-1)


@torch.no_grad()
def heldout_pref_acc(model, tok, rows, block_size, device):
    """held-out 偏好準確率（每 token 平均 logπ：chosen > rejected 的比例）。

    用「每 token 平均」而非「總和」是為了去掉長度偏誤——量的才是內容偏好，不是長度。
    這是 DPO 真正該監控的訊號：train-acc 會衝到 100%，但能不能類推看這條。
    """
    was_training = model.training
    model.eval()
    wins = 0
    for r in rows:
        means = []
        for key in ("chosen", "rejected"):
            ids, m = build_example(tok, r["prompt"], r[key], block_size)
            X = torch.tensor([ids], device=device)
            M = torch.tensor([m], dtype=torch.float, device=device)
            ntok = max(1.0, M[:, 1:].sum().item())
            means.append(seq_logp(model, X, M).item() / ntok)
        wins += int(means[0] > means[1])
    if was_training:
        model.train()
    return wins / len(rows)


def dpo_loss(policy_lp, ref_lp, beta):
    """DPO 損失 + 診斷量。policy_lp/ref_lp：交錯排好的 (chosen, rejected) logπ。"""
    pc, pr = policy_lp[0::2], policy_lp[1::2]     # policy: chosen / rejected
    rc, rr = ref_lp[0::2], ref_lp[1::2]           # reference: chosen / rejected
    # 隱式獎勵 = β·(logπ - logπ_ref)；margin = chosen 獎勵 - rejected 獎勵
    margin = (pc - rc) - (pr - rr)
    loss = -F.logsigmoid(beta * margin).mean()
    acc = (margin > 0).float().mean()             # 有多常「偏好 chosen」
    return loss, margin.mean().item(), acc.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ART / "sft_ckpt.pt"), help="DPO 從 SFT 模型起步")
    ap.add_argument("--dpo_data", default=str(ART / "dpo.jsonl"))
    ap.add_argument("--heldout", default=str(ART / "dpo_heldout.jsonl"))
    ap.add_argument("--out", default=str(ART / "dpo_ckpt.pt"))
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-4)         # DPO 用比 SFT 小的 lr
    ap.add_argument("--beta", type=float, default=0.1)        # KL 約束強度
    ap.add_argument("--batch_size", type=int, default=16)     # 一個 batch 幾組偏好對
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log_csv", default=str(ART / "runs" / "dpo.csv"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tok = load_tokenizer(ART / "tokenizer.json")
    rows = [json.loads(l) for l in
            Path(args.dpo_data).read_text(encoding="utf-8").splitlines()]
    held = []
    hp = Path(args.heldout)
    if hp.exists():
        held = [json.loads(l) for l in hp.read_text(encoding="utf-8").splitlines()]
    print(f"DPO 偏好資料：{len(rows)} 組偏好對（held-out {len(held)} 組監控用）")

    ckpt = torch.load(args.base, map_location=device)
    gcfg = GPTConfig(**ckpt["gpt_config"])
    # policy（可訓練）+ reference（凍結，當錨）都從 SFT 權重複製
    policy = GPT(gcfg).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.train()
    ref = deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(args.seed)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    hist = ["step,loss,margin,train_acc,held_acc"]
    for it in range(args.iters + 1):
        idx = torch.randint(len(rows), (args.batch_size,), generator=g).tolist()
        X, M = collate([rows[i] for i in idx], tok, args.block_size, device)
        policy_lp = seq_logp(policy, X, M)
        with torch.no_grad():
            ref_lp = seq_logp(ref, X, M)
        loss, margin, acc = dpo_loss(policy_lp, ref_lp, args.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 100 == 0:
            # held-out 偏好準確率：DPO 真正該看的「有沒有類推」訊號（train-acc 會騙人）
            hacc = heldout_pref_acc(policy, tok, held, args.block_size, device) if held else float("nan")
            print(f"step {it:5d} | loss {loss.item():.4f} | margin {margin:+7.2f} | "
                  f"train-acc {acc*100:4.0f}% | held-out {hacc*100:4.0f}%")
            hist.append(f"{it},{loss.item():.4f},{margin:.4f},{acc:.4f},{hacc:.4f}")

    Path(args.log_csv).write_text("\n".join(hist) + "\n")
    torch.save({"model": policy.state_dict(), "gpt_config": ckpt["gpt_config"],
                "iter": args.iters, "dpo": True, "beta": args.beta}, args.out)
    print(f"完成 → {args.out}（訓練曲線 → {args.log_csv}）")


if __name__ == "__main__":
    main()
