"""Stage 8 — GRPO（RLHF 第二塊：用 RL 拿 reward model 的分數去優化 policy）。

GRPO（Group Relative Policy Optimization，DeepSeek）＝比 PPO 簡單的現代 RLHF：
**不需要 critic/value 網路**。對每個 prompt 取樣一「組」K 個回答，用 reward model 打分，
把「贏過同組平均多少」當 advantage（組內標準化），再用 policy gradient 更新，外加一個
KL 罰把 policy 拉住、別離 reference(SFT) 太遠。

  loss = − E[ advantage · logπ(回答) ] + β · KL(π ‖ π_ref)
  advantage = (reward − 組平均) / (組標準差)

對照 DPO：DPO 不取樣、不用 RM，一條封閉式損失直接吃靜態偏好對；GRPO 真的「讓模型自己
生成 → 被打分 → 強化高分的、抑制低分的」，是線上的試誤學習（更像 RLHF 本尊，也更會出包）。

**本檔刻意同時量兩把尺**：① RM 給的分數（代理指標，RL 在最大化它）② 真實生成重複率
（獨立指標，RM 沒直接優化它）。兩者背離 = **reward hacking**（policy 鑽 RM 漏洞拿高分、
真實品質沒跟上）＝ RLHF 最有名的坑、也是「指標被優化就失效」(Goodhart) 的活教材。

  python pipeline/08_grpo.py --iters 150 --group 4 --beta 0.05
"""

import argparse
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig          # noqa: E402
from src.model import GPT                  # noqa: E402
from src.reward_model import RewardModel   # noqa: E402
from src.tokenizer import load_tokenizer   # noqa: E402

dpo = importlib.import_module("pipeline.06_dpo")
ART = ROOT / "artifacts"


def load_policy(path, device):
    ck = torch.load(path, map_location=device)
    m = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    m.load_state_dict(ck["model"])
    return m, ck


@torch.no_grad()
def sample_group(policy, prompt_ids, K, max_new, device):
    """同一個 prompt 取樣 K 個回答。回傳 (full: (K, Tp+max_new), Tp)。"""
    ids = torch.tensor([prompt_ids], device=device).repeat(K, 1)
    out = policy.generate(ids, max_new, temperature=1.0, top_p=0.95)
    return out, len(prompt_ids)


def resp_logp(model, X, Tp):
    """teacher-force 算「回答段」每 token logπ 之和。X:(B,L)，回答從位置 Tp 起。"""
    logits, _ = model(X[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)   # (B, L-1)
    mask = torch.zeros_like(tok_logp)
    mask[:, Tp - 1:] = 1.0                       # 目標為回答 token 的位置（含 prompt 末→回答首）
    return (tok_logp * mask).sum(-1), mask.sum(-1)


@torch.no_grad()
def repetition_rate(policy, tok, prompts, max_new, device):
    """真實生成的重複率＝1 − distinct-2（獨立於 RM 的品質尺；高=陷入重複）。"""
    torch.manual_seed(1234)
    rates = []
    for q in prompts:
        ids = torch.tensor([tok.encode(dpo.PROMPT_TMPL.format(q=q))], device=device)
        out = policy.generate(ids, max_new, temperature=0.8, top_p=0.9)[0].tolist()
        gen = out[ids.shape[1]:]
        bg = list(zip(gen, gen[1:]))
        if bg:
            rates.append(1 - len(set(bg)) / len(bg))
    return sum(rates) / len(rates)


@torch.no_grad()
def diversity(policy, tok, prompts, max_new, device):
    """跨 prompt 的輸出多樣性＝不同生成 / 總生成。低=mode collapse（不管問什麼都吐同一句）。

    reward hacking 常見長相：policy 找到一句「RM 給高分的安全答案」，無視問題一直吐它
    → RM 分數高、但其實沒在回答（真實品質崩）。重複率(單句內)抓不到這個，多樣性(跨句)抓得到。
    """
    torch.manual_seed(4321)
    outs = []
    for q in prompts:
        ids = torch.tensor([tok.encode(dpo.PROMPT_TMPL.format(q=q))], device=device)
        out = policy.generate(ids, max_new, temperature=0.8, top_p=0.9)[0].tolist()
        outs.append(tuple(out[ids.shape[1]:]))
    return len(set(outs)) / len(outs)


@torch.no_grad()
def mean_reward(rm, policy, tok, prompts, K, max_new, device):
    """固定 prompts 上的平均 RM 分數（RL 想最大化的代理指標）。"""
    tot, n = 0.0, 0
    for q in prompts:
        pid = tok.encode(dpo.PROMPT_TMPL.format(q=q))
        full, Tp = sample_group(policy, pid, K, max_new, device)
        last = torch.full((full.size(0),), full.size(1) - 1, device=device)
        tot += rm(full, last).mean().item()
        n += 1
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ART / "sft_ckpt.pt"))
    ap.add_argument("--reward", default=str(ART / "reward_ckpt.pt"))
    ap.add_argument("--data", default=str(ART / "dpo_format.jsonl"))
    ap.add_argument("--heldout", default=str(ART / "dpo_format_heldout.jsonl"))
    ap.add_argument("--out", default=str(ART / "grpo_ckpt.pt"))
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--group", type=int, default=4, help="每個 prompt 取樣幾個回答（組大小 K）")
    ap.add_argument("--prompts_per_step", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.05, help="KL 罰：拉住 policy 別離 SFT 太遠")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log_csv", default=str(ART / "runs" / "grpo.csv"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tok = load_tokenizer(ART / "tokenizer.json")
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines()]
    held = [json.loads(l) for l in Path(args.heldout).read_text(encoding="utf-8").splitlines()]
    eval_prompts = [r["prompt"] for r in held[:16]]

    policy, ck = load_policy(args.policy, device)
    policy.train()
    ref = deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rm = RewardModel.from_ckpt(args.reward, device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(args.seed)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    hist = ["step,reward,kl,repetition,diversity"]
    print(f"GRPO：policy←SFT、RM 凍結、K={args.group}、β={args.beta}")
    r0 = mean_reward(rm, policy, tok, eval_prompts, args.group, args.max_new, device)
    rep0 = repetition_rate(policy, tok, eval_prompts, args.max_new, device)
    div0 = diversity(policy, tok, eval_prompts, args.max_new, device)
    print(f"起點：RM 分數 {r0:+.2f}（代理）｜重複率 {rep0*100:.0f}%｜多樣性 {div0*100:.0f}%（真實品質）")

    for it in range(1, args.iters + 1):
        pidx = torch.randint(len(rows), (args.prompts_per_step,), generator=g).tolist()
        seqs, Tps, rewards = [], [], []
        for pi in pidx:
            pid = tok.encode(dpo.PROMPT_TMPL.format(q=rows[pi]["prompt"]))
            full, Tp = sample_group(policy, pid, args.group, args.max_new, device)
            last = torch.full((full.size(0),), full.size(1) - 1, device=device)
            r = rm(full, last)                                   # (K,)
            adv = (r - r.mean()) / (r.std() + 1e-6)              # 組內標準化 advantage
            for k in range(full.size(0)):
                seqs.append(full[k].tolist())
                Tps.append(Tp)
                rewards.append(adv[k].item())

        # 因為各 prompt 長度不同，逐序列算 logπ（小批、tiny model，夠快）
        pg_loss = kl_loss = 0.0
        adv_t = torch.tensor(rewards, device=device)
        for j, s in enumerate(seqs):
            X = torch.tensor([s], device=device)
            lp_pi, ntok = resp_logp(policy, X, Tps[j])
            with torch.no_grad():
                lp_ref, _ = resp_logp(ref, X, Tps[j])
            pg_loss = pg_loss - adv_t[j] * lp_pi                 # −advantage·logπ
            kl_loss = kl_loss + (lp_pi - lp_ref)                 # KL≈ Σ(logπ−logπ_ref)
        loss = (pg_loss + args.beta * kl_loss) / len(seqs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if it % 25 == 0 or it == args.iters:
            rew = mean_reward(rm, policy, tok, eval_prompts, args.group, args.max_new, device)
            rep = repetition_rate(policy, tok, eval_prompts, args.max_new, device)
            div = diversity(policy, tok, eval_prompts, args.max_new, device)
            kl = (kl_loss / len(seqs)).item()
            print(f"step {it:4d} | RM 分數 {rew:+.2f} | KL {kl:+.2f} | "
                  f"重複率 {rep*100:4.0f}% | 多樣性 {div*100:4.0f}%")
            hist.append(f"{it},{rew:.4f},{kl:.4f},{rep:.4f},{div:.4f}")

    Path(args.log_csv).write_text("\n".join(hist) + "\n")
    torch.save({"model": policy.state_dict(), "gpt_config": ck["gpt_config"], "grpo": True}, args.out)
    print(f"完成 → {args.out}（曲線 → {args.log_csv}）")


if __name__ == "__main__":
    main()
