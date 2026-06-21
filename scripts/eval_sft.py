"""SFT 專用評估：在「沒訓過的 held-out 指令」上比 base vs SFT。

關鍵：不能用維基 perplexity（那是預訓練的尺，SFT 會被 alignment tax 拖低）。
SFT 的尺是「按指令應答的能力」，這裡量兩個：
- 回答段 perplexity：只算「答案 token」的 loss（給定問題，模型多會產生正確答案）。SFT 該更低。
- win-rate：逐題比，SFT 對 gold 答案 loss 較低就算「贏」（免 LLM 評審的客觀代理）。

  python scripts/eval_sft.py
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import GPTConfig          # noqa: E402
from src.model import GPT                  # noqa: E402
from src.tokenizer import load_tokenizer   # noqa: E402

ART = ROOT / "artifacts"


def load(p, device):
    ck = torch.load(p, map_location=device)
    m = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def resp_loss(model, tok, q, a, device):
    """只在『答案 token』上算 CE loss（指令段不算）——這才是 SFT 在乎的。"""
    prompt = tok.encode(f"問：{q}\n答：")
    ids = prompt + tok.encode(a)
    if len(ids) < len(prompt) + 1:
        return None
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    with torch.no_grad():
        logits, _ = model(x)
    start = len(prompt) - 1                          # y 裡答案 token 的起點
    return F.cross_entropy(logits[0, start:], y[0, start:]).item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(ART / "tokenizer.json")
    base = load(ART / "ckpt.pt", device)
    sft = load(ART / "sft_ckpt.pt", device)
    held = [json.loads(line) for line in
            (ART / "sft_heldout.jsonl").read_text(encoding="utf-8").splitlines()]

    def answers_in_form(model, q):
        """生成式行為指標：回答段是否「以定義句開頭」（X是… / 是一…），而非亂續寫。"""
        ids = torch.tensor([tok.encode(f"問：{q}\n答：")], device=device)
        with torch.no_grad():
            out = model.generate(ids, 18, temperature=0.5, top_p=0.9)[0].tolist()
        ans = tok.decode(out).split("答：", 1)[-1][:14]
        return ("是一" in ans) or (ans[:1] and ans[:1] in q and "是" in ans[:6])

    bl = sl = 0.0
    wins = n = 0
    bform = sform = 0
    for p in held:
        lb = resp_loss(base, tok, p["q"], p["a"], device)
        ls = resp_loss(sft, tok, p["q"], p["a"], device)
        if lb is None or ls is None:
            continue
        bl += lb
        sl += ls
        wins += int(ls < lb)
        n += 1
        bform += int(answers_in_form(base, p["q"]))
        sform += int(answers_in_form(sft, p["q"]))

    print(f"held-out 指令（SFT 沒看過）：{n} 筆")
    print("=" * 56)
    print("尺A｜回答段 perplexity（對 gold 答案）— ⚠️ 被汙染")
    print(f"  base {math.exp(bl/n):6.1f}   SFT {math.exp(sl/n):6.1f}   → base 較低")
    print("  但 gold 答案=維基定義句，base 預訓練看過 → 不公平；SFT 的對齊稅讓它對特定 gold 變差。")
    print("-" * 56)
    print("尺B｜應答行為（生成是否以定義句應答，不靠 gold）— 公平")
    print(f"  base {bform/n*100:3.0f}%   SFT {sform/n*100:3.0f}%   → SFT 該明顯高")
    print("=" * 56)
    print("結論：選錯尺（A）會說 SFT 變爛；選對尺（B，量『行為』不量『背特定答案』）才看出 SFT 的價值。")


if __name__ == "__main__":
    main()
