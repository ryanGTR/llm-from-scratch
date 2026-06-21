"""Stage 7 — 訓練 Reward Model（RLHF 第一塊）。

從 SFT 接骨幹，加一個純量 head，用偏好對（chosen 該比 rejected 高分）以 Bradley-Terry
損失訓練。訓好的 RM 之後給 Stage 8 的 GRPO 當「分數來源」。

用 format 軸偏好（連貫 vs 退化重複，模型學得動）——RM 學「連貫的得高分、退化重複的得低分」。

  python pipeline/07_reward_model.py --iters 400
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reward_model import RewardModel, bt_loss   # noqa: E402
from src.tokenizer import load_tokenizer            # noqa: E402

dpo = importlib.import_module("pipeline.06_dpo")     # 重用 build_example / PROMPT_TMPL
ART = ROOT / "artifacts"


def collate_rewards(rows, key, tok, block_size, device):
    """把一批回答編成 (X, last_idx)：X 右側 padding、last_idx 指到最後一個回答 token。"""
    seqs = []
    for r in rows:
        ids, _ = dpo.build_example(tok, r["prompt"], r[key], block_size)
        seqs.append(ids)
    maxlen = max(len(s) for s in seqs)
    X = torch.zeros(len(seqs), maxlen, dtype=torch.long, device=device)
    last = torch.zeros(len(seqs), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor(s, device=device)
        last[i] = len(s) - 1                  # 最後一個真實 token（不是 pad）
    return X, last


@torch.no_grad()
def heldout_acc(rm, tok, rows, block_size, device):
    Xc, lc = collate_rewards(rows, "chosen", tok, block_size, device)
    Xr, lr = collate_rewards(rows, "rejected", tok, block_size, device)
    rc, rr = rm(Xc, lc), rm(Xr, lr)
    return (rc > rr).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default=str(ART / "sft_ckpt.pt"))
    ap.add_argument("--data", default=str(ART / "dpo_format.jsonl"))
    ap.add_argument("--heldout", default=str(ART / "dpo_format_heldout.jsonl"))
    ap.add_argument("--out", default=str(ART / "reward_ckpt.pt"))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tok = load_tokenizer(ART / "tokenizer.json")
    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines()]
    held = [json.loads(l) for l in Path(args.heldout).read_text(encoding="utf-8").splitlines()]
    print(f"RM 訓練偏好對：{len(rows)}（held-out {len(held)}）")

    ck = torch.load(args.sft, map_location=device)
    rm = RewardModel(ck["gpt_config"]).to(device)
    rm.gpt.load_state_dict(ck["model"])               # 骨幹接 SFT；v_head 隨機初始化
    rm.train()
    opt = torch.optim.AdamW(rm.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(args.seed)
    for it in range(args.iters + 1):
        idx = torch.randint(len(rows), (args.batch_size,), generator=g).tolist()
        batch = [rows[i] for i in idx]
        Xc, lc = collate_rewards(batch, "chosen", tok, args.block_size, device)
        Xr, lr = collate_rewards(batch, "rejected", tok, args.block_size, device)
        loss, acc = bt_loss(rm(Xc, lc), rm(Xr, lr))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 100 == 0:
            hacc = heldout_acc(rm, tok, held, args.block_size, device)
            print(f"step {it:4d} | bt_loss {loss.item():.4f} | "
                  f"train-acc {acc.item()*100:4.0f}% | held-out {hacc*100:4.0f}%")

    torch.save({"model": rm.state_dict(), "gpt_config": ck["gpt_config"]}, args.out)
    print(f"完成 → {args.out}")


if __name__ == "__main__":
    main()
