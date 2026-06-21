"""比較兩個模型的「產出」，不是只比延遲。

金絲雀放量前、壓縮上線前，真正要問的是「新模型答得一樣好/一樣嗎」。給兩個指標：
- test_loss：各自在同一份 test 集的損失（誰準）。
- agreement：同樣輸入下，兩模型「下一個字預測相同」的比例（行為多接近）。
  agreement 高 = 候選跟現行幾乎一樣（放量安全）；低 = 行為差很多（要謹慎/人工審）。

  python scripts/compare_models.py artifacts/ckpt.pt /tmp/candidate.pt
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig, TrainConfig   # noqa: E402
from src.model import GPT                        # noqa: E402
from pipeline.train_utils import get_batch       # noqa: E402

ART = ROOT / "artifacts"


def load(path, device):
    ck = torch.load(path, map_location=device)
    m = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def main():
    a_path = sys.argv[1] if len(sys.argv) > 1 else str(ART / "ckpt.pt")
    b_path = sys.argv[2] if len(sys.argv) > 2 else str(ART / "ckpt.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ma, mb = load(a_path, device), load(b_path, device)
    cfg = GPTConfig(**torch.load(a_path, map_location=device)["gpt_config"])

    meta = json.loads((ART / "meta.json").read_text())
    dt = np.dtype(meta.get("token_dtype", "uint16"))
    test = np.fromfile(ART / "test.bin", dtype=dt)

    tcfg = TrainConfig()
    la = lb = 0.0
    agree = tot = 0
    torch.manual_seed(0)                         # 兩模型吃同一批資料才公平
    with torch.no_grad():
        for _ in range(80):
            x, y = get_batch(test, cfg.block_size, tcfg.batch_size, device)
            lga, lossa = ma(x, y)
            lgb, lossb = mb(x, y)
            la += lossa.item()
            lb += lossb.item()
            agree += (lga.argmax(-1) == lgb.argmax(-1)).sum().item()
            tot += lga.argmax(-1).numel()
    la, lb, ag = la / 80, lb / 80, agree / tot

    print(f"A = {a_path}")
    print(f"B = {b_path}")
    print("-" * 50)
    print(f"A test_loss = {la:.4f}  (ppl {math.exp(la):.1f})")
    print(f"B test_loss = {lb:.4f}  (ppl {math.exp(lb):.1f})")
    print(f"Δloss(B-A)  = {lb - la:+.4f}  → {'B 較差' if lb > la + 0.02 else ('B 較好' if lb < la - 0.02 else '幾乎一樣')}")
    print(f"agreement   = {ag*100:.1f}%  (兩模型 next-token 預測相同的比例)")


if __name__ == "__main__":
    main()
