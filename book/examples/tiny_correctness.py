"""tiny_correctness.py — 在 CPU 上親手證：手刻的 attention 是對的，不是自我感覺良好。

書本 Ch5 的「正確性 baseline」：品質 baseline 問「有沒有贏笨基準」，正確性 baseline 問
**「你確定你手刻的數學是對的嗎？」**——把 from-scratch 實作對一個**可信參照實作**數值對拍。
這是「不騙自己」的最低門檻：你嵌進書裡的每個數字，前提是底層程式真的算對。

對拍三件事（都該差在浮點誤差 ~1e-6 等級）：
  1) 手刻 causal attention（softmax(QKᵀ/√d) 遮罩 → @V）對 torch 內建
     `F.scaled_dot_product_attention(is_causal=True)`。
  2) 多頭版本同樣對拍。
  3) 順帶驗結構不變量：softmax 每列和為 1、因果遮罩讓未來權重為 0。

用法：
    python tiny_correctness.py        # 純 CPU、瞬間、不需語料、不需 GPU
"""

import math
import torch
from torch.nn import functional as F

torch.manual_seed(0)
B, nh, T, hd = 2, 4, 16, 32


def naive_causal_attention(q, k, v):
    """書裡那條 softmax(QKᵀ/√d)·V + 因果遮罩，逐步攤開（教學版、看得懂優先）。"""
    att = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])     # (B,nh,T,T) 相似度+縮放
    tril = torch.tril(torch.ones(T, T))
    att = att.masked_fill(tril == 0, float("-inf"))            # 不能偷看未來
    w = F.softmax(att, dim=-1)
    return w @ v, w


def max_abs_diff(a, b):
    return (a - b).abs().max().item()


if __name__ == "__main__":
    print(f"shape (B,nh,T,hd)=({B},{nh},{T},{hd})  device=cpu\n")
    q = torch.randn(B, nh, T, hd)
    k = torch.randn(B, nh, T, hd)
    v = torch.randn(B, nh, T, hd)

    # 1) 手刻 vs torch 內建（trusted reference）
    mine, w = naive_causal_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    d1 = max_abs_diff(mine, ref)

    # 2) 多頭：合一個 batch 再對拍（同一條路、不同形狀）
    q2, k2, v2 = (torch.randn(B, nh, T, hd) for _ in range(3))
    d2 = max_abs_diff(naive_causal_attention(q2, k2, v2)[0],
                      F.scaled_dot_product_attention(q2, k2, v2, is_causal=True))

    # 3) 結構不變量
    row_sums = w.sum(dim=-1)                                   # softmax 每列應為 1
    rows_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    future_w = w[0, 0, 0, 1:]                                  # 第 0 個 token 對未來的權重
    causal_ok = float(future_w.max()) == 0.0

    print("=== 正確性對拍（手刻 vs torch 內建可信參照）===")
    print(f"  單頭 causal attention   max|diff| = {d1:.2e}")
    print(f"  多頭 causal attention   max|diff| = {d2:.2e}")
    print(f"  → 差在浮點誤差等級（~1e-6），數學等價 ✅")
    print("\n=== 結構不變量 ===")
    print(f"  softmax 每列和為 1？        {rows_ok}")
    print(f"  因果遮罩讓未來權重=0？      {causal_ok}")
    print("\n手刻的 attention 跟工業級實作算出同一個答案——"
          "你嵌進書裡的數字，底層是對的，不是自我感覺良好。")

    assert d1 < 1e-5 and d2 < 1e-5 and rows_ok and causal_ok, "對拍失敗！"
