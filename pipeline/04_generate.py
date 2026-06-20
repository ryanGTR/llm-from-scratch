"""Stage 4 — Generate：載入 checkpoint，給 prompt 讓模型續寫。

pipeline 最後一棒，等同「部署後拿來推論」。
用法：python pipeline/04_generate.py --prompt "ROMEO:" --max_new_tokens 300
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig  # noqa: E402
from src.model import GPT  # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    art = Path(args.artifacts)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(art / "tokenizer.json")
    ckpt = torch.load(art / "ckpt.pt", map_location=device)
    gcfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(gcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    start = torch.tensor(
        [tok.encode(args.prompt)], dtype=torch.long, device=device
    )
    out = model.generate(
        start, args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
