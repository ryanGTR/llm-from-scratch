"""Stage 2 — Train：讀 token bin -> 訓練 GPT -> 存 checkpoint。

pipeline 核心一棒。產物：artifacts/ckpt.pt（模型權重 + config）。
自動偵測 CUDA；在 FW16 上請用 `prime-run` 包起來跑才會吃到 NVIDIA 5070。

用法：prime-run python -m pipeline.02_train --max_iters 2000
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig, TrainConfig  # noqa: E402
from src.model import GPT  # noqa: E402
from pipeline.train_utils import get_batch  # noqa: E402


@torch.no_grad()
def estimate_loss(model, splits, tcfg, gcfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(data, gcfg.block_size, tcfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--run_name", default="default",
                    help="這次訓練的名字；loss 記到 artifacts/runs/<name>.csv，方便多次比較")
    ap.add_argument("--use_rmsnorm", action="store_true",
                    help="用 RMSNorm（現代）取代 LayerNorm")
    args = ap.parse_args()

    tcfg = TrainConfig()
    gcfg = GPTConfig()
    # 命令列覆寫（沒給就用 config.py 預設）
    if args.max_iters is not None:
        tcfg.max_iters = args.max_iters
    if args.block_size is not None:
        gcfg.block_size = args.block_size
    if args.n_layer is not None:
        gcfg.n_layer = args.n_layer
    if args.n_embd is not None:
        gcfg.n_embd = args.n_embd
    gcfg.use_rmsnorm = args.use_rmsnorm

    torch.manual_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    art = Path(args.artifacts)
    meta = json.loads((art / "meta.json").read_text())
    gcfg.vocab_size = meta["vocab_size"]
    train_data = np.fromfile(art / "train.bin", dtype=np.uint16)
    val_data = np.fromfile(art / "val.bin", dtype=np.uint16)
    splits = {"train": train_data, "val": val_data}

    model = GPT(gcfg).to(device)
    print(f"參數量 ≈ {model.num_params() / 1e6:.2f} M")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay
    )

    # 監控：把每次 eval 的 loss 記成 CSV，之後 plot_loss.py 畫成曲線、比較不同 run
    runs_dir = art / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    # 每個 run 存一份 meta（跨 tokenizer 比較要用 chars_per_token 算 BPC）
    (runs_dir / f"{args.run_name}.meta.json").write_text(json.dumps({
        "run_name": args.run_name,
        "tokenizer": meta.get("tokenizer", "char"),
        "vocab_size": gcfg.vocab_size,
        "chars_per_token": meta.get("chars_per_token", 1.0),
        "block_size": gcfg.block_size,
        "n_layer": gcfg.n_layer,
        "n_embd": gcfg.n_embd,
    }, indent=2))
    log_path = runs_dir / f"{args.run_name}.csv"
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["step", "train_loss", "val_loss"])

    best_val = float("inf")
    for it in range(tcfg.max_iters + 1):
        if it % tcfg.eval_interval == 0:
            losses = estimate_loss(model, splits, tcfg, gcfg, device)
            print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            logger.writerow([it, f"{losses['train']:.6f}", f"{losses['val']:.6f}"])
            log_file.flush()       # 即時落盤，邊訓邊能看
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "gpt_config": gcfg.__dict__,
                    "val_loss": best_val,
                    "iter": it,
                }, art / "ckpt.pt")

        x, y = get_batch(train_data, gcfg.block_size, tcfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

    log_file.close()
    print(f"完成。best val loss = {best_val:.4f} -> {art}/ckpt.pt")
    print(f"loss 紀錄 -> {log_path}（用 `make plot-loss` 畫曲線）")


if __name__ == "__main__":
    main()
