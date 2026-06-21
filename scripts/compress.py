"""模型壓縮：量化（quantization）——用更少的位元存權重，省記憶體/頻寬，盡量不掉準。

量「省多少 vs 掉多少準」的取捨（又是「準 vs 省」）：
- fp32 → fp16：每個權重 32→16 bit，大小減半，GPU 上幾乎不掉準（業界推論預設）。
- fp32 → int8（dynamic quant）：Linear 層權重壓到 8 bit，CPU 推論用；embedding 不動，
  故 vocab 大的小模型總縮減有限（大模型 Linear 佔比高 → 省更多）。

註：這是「權重量化」。前面聊的 Google TurboQuant 是「KV-cache 量化」——不同軸，可疊加。

  python scripts/compress.py
"""

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig, TrainConfig   # noqa: E402
from src.model import GPT                        # noqa: E402
from pipeline.train_utils import get_batch       # noqa: E402

ART = ROOT / "artifacts"


def serialized_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1e6


def eval_loss(model, data, cfg, n=60) -> float:
    tcfg = TrainConfig()
    tot = 0.0
    with torch.no_grad():
        for _ in range(n):
            x, y = get_batch(data, cfg.block_size, tcfg.batch_size, "cpu")
            _, loss = model(x, y)
            tot += loss.item()
    return tot / n


def main():
    ckpt = torch.load(ART / "ckpt.pt", map_location="cpu")
    cfg = GPTConfig(**ckpt["gpt_config"])
    meta = json.loads((ART / "meta.json").read_text())
    dt = np.dtype(meta.get("token_dtype", "uint16"))
    test = np.fromfile(ART / "test.bin", dtype=dt)

    def fresh():
        m = GPT(cfg)
        m.load_state_dict(ckpt["model"])
        return m.eval()

    # fp32 基準（CPU，與 int8 公平比較）
    m32 = fresh()
    s32, l32 = serialized_mb(m32), eval_loss(m32, test, cfg)

    # int8 dynamic quantization（Linear 層 → 8 bit，CPU 推論）
    m8 = torch.quantization.quantize_dynamic(fresh(), {nn.Linear}, dtype=torch.qint8)
    s8, l8 = serialized_mb(m8), eval_loss(m8, test, cfg)

    # fp16：真的轉半精度再序列化量大小（near-lossless，GPU 推論預設；CPU fp16 算 loss 不穩故略）
    s16 = serialized_mb(fresh().half())

    print(f"{'精度':6} {'大小(MB)':>10} {'壓縮':>7} {'test_loss':>10} {'ppl':>7} {'Δloss':>7}")
    print("-" * 56)
    print(f"{'fp32':6} {s32:>10.2f} {'1.00x':>7} {l32:>10.4f} {math.exp(l32):>7.1f} {'(基準)':>7}")
    print(f"{'fp16':6} {s16:>10.2f} {s32/s16:>6.2f}x {'~ 同 fp32':>10} {'—':>7} {'~0':>7}")
    print(f"{'int8':6} {s8:>10.2f} {s32/s8:>6.2f}x {l8:>10.4f} {math.exp(l8):>7.1f} {l8-l32:>+7.4f}")
    print("-" * 56)
    print(f"int8：大小縮 {(1-s8/s32)*100:.0f}%、test_loss 變動 {l8-l32:+.4f}"
          f"（embedding 沒量化，vocab={cfg.vocab_size} 佔比高所以總縮減有限）")


if __name__ == "__main__":
    main()
