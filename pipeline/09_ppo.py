r"""Stage 9 — PPO（補完 RL 家族：看 GRPO 到底簡化掉了什麼）。

PPO（Schulman 2017）是 InstructGPT/ChatGPT 用的經典 RLHF 演算法。和我們做過的 GRPO 比，
PPO 多了**兩個招牌零件**：

1. **Critic / value 網路 $V_\phi$**：學一個「這個 prompt 大概能拿多少獎勵」的基準，
   用 advantage $A=R-V(x)$ 取代 reward 本身降變異數。**GRPO 把它丟了**，改用「同組 K 個回答的
   平均獎勵」當基準（見 derivations §4：組內平均也是不偏 baseline）。所以 GRPO 不需要 critic。
2. **Clipped surrogate（截斷目標）**：用 importance ratio $\rho=\pi_\theta/\pi_{\theta_\text{old}}$ 重複利用同一批
   rollout 訓好幾個 epoch，並把 $\rho$ 夾在 $[1-\epsilon,1+\epsilon]$ 內 → **限制單次更新幅度**，防一步走太遠把
   policy 走壞。這是 PPO 名字裡「Proximal（近端）」的由來。

本檔刻意做 **clip vs 無 clip 對照**：無 clip 時多 epoch 重用 rollout 會讓 ratio 失控、KL 爆掉
（destructive update）；clip 把它夾住 → 穩。這就是 PPO 招牌機制的價值。

  python pipeline/09_ppo.py --iters 120 --clip_eps 0.2          # 有 clip（穩）
  python pipeline/09_ppo.py --iters 120 --clip_eps 0            # 無 clip（會失控）
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
grpo = importlib.import_module("pipeline.08_grpo")
ART = ROOT / "artifacts"


def load_policy(path, device):
    ck = torch.load(path, map_location=device)
    m = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    m.load_state_dict(ck["model"])
    return m, ck


def ppo_policy_loss(ratio, adv, resp_mask, clip_eps):
    """PPO clipped surrogate（per token）。clip_eps<=0 → 不截斷（對照組）。回傳 (loss, clip_frac)。

    adv>0 時若 ratio 衝過 1+ε，min 會選到被夾住的那項 → 限制這步能拿多少 policy gain
    → 防一步走太遠。這就是 PPO 的「Proximal」。
    """
    a = adv.unsqueeze(1) if adv.dim() == 1 else adv
    surr1 = ratio * a
    if clip_eps > 0:
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
        term = torch.min(surr1, surr2)
        clipped = ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float()
    else:
        term = surr1
        clipped = torch.zeros_like(ratio)
    denom = resp_mask.sum().clamp(min=1.0)
    loss = -(term * resp_mask).sum() / denom
    return loss, (clipped * resp_mask).sum().item() / denom.item()


def token_logp(model, X):
    """每 token 的 logπ（不加總）。回傳 [B, L-1]，對齊到「被預測的那個 token」。"""
    logits, _ = model(X[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout(policy, rm, tok, prompts, max_new, device):
    """取樣一批 (prompt→response)，打包成右側 padding 的張量 + 每序列的 reward / prompt 長度。"""
    seqs, Tps = [], []
    for q in prompts:
        pid = tok.encode(dpo.PROMPT_TMPL.format(q=q))
        ids = torch.tensor([pid], device=device)
        out = policy.generate(ids, max_new, temperature=1.0, top_p=0.95)[0].tolist()
        seqs.append(out)
        Tps.append(len(pid))
    L = max(len(s) for s in seqs)
    X = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
    resp_mask = torch.zeros(len(seqs), L - 1, device=device)   # 對齊到 target frame
    last = torch.zeros(len(seqs), dtype=torch.long, device=device)
    plast = torch.zeros(len(seqs), dtype=torch.long, device=device)
    for i, (s, Tp) in enumerate(zip(seqs, Tps)):
        X[i, :len(s)] = torch.tensor(s, device=device)
        resp_mask[i, Tp - 1:len(s) - 1] = 1.0      # 回答段 token 的位置
        last[i] = len(s) - 1                        # 最後一個回答 token（給 RM 打分）
        plast[i] = Tp - 1                           # prompt 最後一個 token（給 critic 估值）
    R = rm(X, last)                                 # [B] reward
    return X, resp_mask, last, plast, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ART / "sft_ckpt.pt"))
    ap.add_argument("--reward", default=str(ART / "reward_ckpt.pt"))
    ap.add_argument("--data", default=str(ART / "dpo_format.jsonl"))
    ap.add_argument("--heldout", default=str(ART / "dpo_format_heldout.jsonl"))
    ap.add_argument("--out", default=str(ART / "ppo_ckpt.pt"))
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--prompts_per_step", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4, help="同一批 rollout 重用幾個 epoch（PPO 的精神）")
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip_eps", type=float, default=0.2, help="ratio 截斷範圍；設 0 = 不截斷（對照組）")
    ap.add_argument("--beta", type=float, default=0.02, help="KL-to-ref 罰")
    ap.add_argument("--vcoef", type=float, default=0.5, help="value loss 權重")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log_csv", default=str(ART / "runs" / "ppo.csv"))
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
    critic = RewardModel(ck["gpt_config"]).to(device)   # value 網路：GPT 骨幹 + 純量 head
    critic.gpt.load_state_dict(ck["model"])             # 從 SFT 接骨幹
    critic.train()
    opt = torch.optim.AdamW(list(policy.parameters()) + list(critic.parameters()), lr=args.lr)

    use_clip = args.clip_eps > 0
    print(f"PPO：critic=on、epochs={args.epochs}、clip={'ε='+str(args.clip_eps) if use_clip else 'OFF（對照）'}")
    g = torch.Generator().manual_seed(args.seed)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    hist = ["step,reward,value_loss,kl,clip_frac,diversity"]
    r0 = grpo.mean_reward(rm, policy, tok, eval_prompts, 4, args.max_new, device)
    print(f"起點：RM 分數 {r0:+.2f}")

    for it in range(1, args.iters + 1):
        pidx = torch.randint(len(rows), (args.prompts_per_step,), generator=g).tolist()
        prompts = [rows[i]["prompt"] for i in pidx]
        X, resp_mask, last, plast, R = rollout(policy, rm, tok, prompts, args.max_new, device)
        with torch.no_grad():
            logp_old = token_logp(policy, X)            # rollout 當下的 logπ_old（固定）
            logp_ref = token_logp(ref, X)
        denom = resp_mask.sum().clamp(min=1.0)

        vloss_log = kl_log = clipf_log = 0.0
        for _ in range(args.epochs):
            V = critic(X, plast)                        # 估值 V(x)（讀 prompt 末 token，因果故看不到回答）
            adv = (R - V.detach())
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)   # 標準化 advantage（PPO 慣例）
            logp_new = token_logp(policy, X)
            ratio = torch.exp(logp_new - logp_old)      # importance ratio（per token）
            pol_loss, clipf_log = ppo_policy_loss(ratio, adv, resp_mask, args.clip_eps)
            kl = ((logp_new - logp_ref) * resp_mask).sum() / denom
            val_loss = ((V - R) ** 2).mean()            # critic：回歸到真實 reward
            loss = pol_loss + args.beta * kl + args.vcoef * val_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 1.0)
            opt.step()
            vloss_log, kl_log = val_loss.item(), kl.item()

        if it % 20 == 0 or it == args.iters:
            rew = grpo.mean_reward(rm, policy, tok, eval_prompts, 4, args.max_new, device)
            div = grpo.diversity(policy, tok, eval_prompts, args.max_new, device)
            print(f"step {it:4d} | RM {rew:+.2f} | value_loss {vloss_log:6.2f} | "
                  f"KL {kl_log:+7.2f} | clip% {clipf_log*100:3.0f} | 多樣性 {div*100:3.0f}%")
            hist.append(f"{it},{rew:.4f},{vloss_log:.4f},{kl_log:.4f},{clipf_log:.4f},{div:.4f}")

    Path(args.log_csv).write_text("\n".join(hist) + "\n")
    torch.save({"model": policy.state_dict(), "gpt_config": ck["gpt_config"], "ppo": True}, args.out)
    print(f"完成 → {args.out}（曲線 → {args.log_csv}）")


if __name__ == "__main__":
    main()
