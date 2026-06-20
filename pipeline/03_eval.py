"""Stage 3 — Eval：載入 checkpoint，算 val loss 與 perplexity。

pipeline 的「品質閘門」。真實 MLOps 會在這裡判斷模型夠不夠好、要不要
往下游部署（就像 CI 的測試 gate）。這裡先給最基本的指標。

用法：python -m pipeline.03_eval
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig, TrainConfig  # noqa: E402
from src.model import GPT  # noqa: E402
from pipeline.train_utils import get_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="在哪個切分上評估；test=訓練/驗證都沒碰過的最終成績")
    args = ap.parse_args()

    art = Path(args.artifacts)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(art / "ckpt.pt", map_location=device)
    gcfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(gcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = json.loads((art / "meta.json").read_text())
    dt = np.dtype(meta.get("token_dtype", "uint16"))
    data = np.fromfile(art / f"{args.split}.bin", dtype=dt)
    tcfg = TrainConfig()

    losses = torch.zeros(args.eval_iters)
    with torch.no_grad():
        for k in range(args.eval_iters):
            x, y = get_batch(data, gcfg.block_size, tcfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()

    eval_loss = losses.mean().item()
    ppl = math.exp(eval_loss)
    report = {
        "split": args.split,
        f"{args.split}_loss": round(eval_loss, 4),
        "perplexity": round(ppl, 2),
        "train_iter": ckpt.get("iter"),
        "params_M": round(model.num_params() / 1e6, 3),
    }
    print(json.dumps(report, indent=2))
    (art / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
