"""把訓練好的模型對一段文字的 attention 畫成熱圖。

熱圖怎麼看：
  - 直軸（列）= 正在「看」的 token（query，第 i 個字）
  - 橫軸（行）= 被「看」的 token（key，第 j 個字）
  - 顏色越亮 = 第 i 個字越注意第 j 個字
  - 只有左下三角有值（因果遮罩：不能看未來）

用法：python pipeline/viz_attention.py --layer 0 --head 0 --text "ROMEO: But soft"
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import torch                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig   # noqa: E402
from src.model import GPT          # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402
from src.viz import get_attention  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--text", default="ROMEO: But soft, what")
    ap.add_argument("--layer", type=int, default=0, help="第幾層（0 起）")
    ap.add_argument("--head", type=int, default=0, help="第幾個頭（0 起）")
    ap.add_argument("--grid", action="store_true",
                    help="一次畫全部 layer×head，看不同頭的分工")
    args = ap.parse_args()

    art = Path(args.artifacts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer(art / "tokenizer.json")
    ckpt = torch.load(art / "ckpt.pt", map_location=device)
    gcfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(gcfg).to(device)
    model.load_state_dict(ckpt["model"])

    ids = tok.encode(args.text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    atts = get_attention(model, idx)            # list[n_layer] of (n_head,T,T)

    if args.grid:
        # 全部 layer×head 一次畫，看不同頭學到什麼不同的關注模式
        nl, nh = gcfg.n_layer, gcfg.n_head
        fig, axes = plt.subplots(nl, nh, figsize=(nh * 2.2, nl * 2.2))
        for l in range(nl):
            for h in range(nh):
                ax = axes[l][h] if nl > 1 else axes[h]
                ax.imshow(atts[l][h].numpy(), cmap="viridis", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
                if l == 0:
                    ax.set_title(f"head {h}", fontsize=9)
                if h == 0:
                    ax.set_ylabel(f"layer {l}", fontsize=9)
        fig.suptitle(f"全部 {nl}×{nh} 個 attention head（文字：{args.text!r}）"
                     f"｜亮=注意、左下三角=因果", fontsize=11)
        out = art / "attention_grid.png"
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"已存：{out}（{nl} 層 × {nh} 頭 = {nl*nh} 張）")
        print(f"文字：{args.text!r}")
        return

    att = atts[args.layer][args.head].numpy()   # (T, T)

    labels = [c if c != "\n" else "\\n" for c in args.text]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5),
                                    max(5, len(labels) * 0.5)))
    im = ax.imshow(att, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("被注意的字（key）")
    ax.set_ylabel("正在看的字（query）")
    ax.set_title(f"Attention 熱圖｜layer {args.layer} head {args.head}"
                 f"（n_layer={gcfg.n_layer}, n_head={gcfg.n_head}）")
    fig.colorbar(im, ax=ax, label="注意力權重（每列加總=1）")
    out = art / "attention.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"已存：{out}")
    print(f"文字：{args.text!r}  layer={args.layer} head={args.head}")


if __name__ == "__main__":
    main()
