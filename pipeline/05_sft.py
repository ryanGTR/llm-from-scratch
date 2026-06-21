"""Stage 5 — SFT（指令微調）：後訓練的第一步，把 base 模型教成「會聽話的對話格式」。

從現行 production base 接著訓，但餵的是「問：…\n答：…」的對話格式資料（scripts/make_sft_data.py
產的）。base 的 loss 算在整段格式上，模型學會「看到『答：』就在那位置產生回應」。

這不會讓小模型變聰明（規模不夠），是示範 SFT 的「機制」：行為從「續寫維基」變成「應答格式」。
註：正式 SFT 會「只在回答段算 loss（mask 掉指令）」，這裡為了最小可見先算整段。

  python pipeline/05_sft.py --iters 1500
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig, TrainConfig   # noqa: E402
from src.model import GPT                        # noqa: E402
from src.tokenizer import load_tokenizer         # noqa: E402
from pipeline.train_utils import get_batch       # noqa: E402

ART = Path("artifacts")
TEMPLATE = "問：{q}\n答：{a}\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ART / "ckpt.pt"), help="base 模型（現行 production）")
    ap.add_argument("--sft_data", default=str(ART / "sft.jsonl"))
    ap.add_argument("--out", default=str(ART / "sft_ckpt.pt"))
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--block_size", type=int, default=128)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = TrainConfig()

    tok = load_tokenizer(ART / "tokenizer.json")
    pairs = [json.loads(line) for line in Path(args.sft_data).read_text(encoding="utf-8").splitlines()]
    text = "".join(TEMPLATE.format(q=p["q"], a=p["a"]) for p in pairs)
    data = np.array(tok.encode(text), dtype=np.uint16)
    print(f"SFT 資料：{len(pairs)} 配對 → {len(data):,} token")

    ckpt = torch.load(args.base, map_location=device)
    gcfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(gcfg).to(device)
    model.load_state_dict(ckpt["model"])          # 從 base 接著訓
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for it in range(args.iters + 1):
        x, y = get_batch(data, args.block_size, tcfg.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 200 == 0:
            print(f"step {it:5d} | sft_loss {loss.item():.4f}")

    torch.save({"model": model.state_dict(), "gpt_config": ckpt["gpt_config"],
                "iter": args.iters, "sft": True}, args.out)
    print(f"完成 → {args.out}")


if __name__ == "__main__":
    main()
