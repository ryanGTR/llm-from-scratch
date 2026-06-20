---
title: 現代化零件 — 把 GPT-2 配方升級成 LLaMA 配方
type: retrospective
created: 2026-06-20
updated: 2026-06-20
tags: [llm, transformer, modernization, rmsnorm, swiglu, rope, gqa]
sources: [src/model.py, src/config.py, pipeline/02_train.py]
---

# 現代化零件總回顧

把 2017 原版 Transformer 的四個零件，逐一換成 2023 主流（LLaMA/Mistral 同款）。
方法：每個零件「先預測 → 實測 → 看 loss 曲線 → 加測試 → commit」。全部做成
`config` 開關，可一鍵在「GPT-2 配方 ↔ LLaMA 配方」之間切換對比。

## 四個零件

| 零件 | 換掉什麼 | 口味 | best val | vs 原版 1.7712 | 備註 |
|---|---|---|---|---|---|
| **RMSNorm** | LayerNorm | 省 | 1.7724 | ≈ 平手 | 砍掉減均值+bias，更簡單 |
| **SwiGLU** | GELU MLP | 準 | 1.6810 | **−0.09** | gating 提升表達力，同參數 |
| **RoPE** | 學習式位置 embedding | 準 + 外推 | 1.6248 | **−0.146** | 編碼相對距離；參數更少；能外推 |
| **GQA** | 標準 MHA | 省 | 1.7974 | +0.026 | KV-cache 砍半（512 vs 1024）|

## 完全體（四個全開）

| 模型 | best val | 參數 |
|---|---|---|
| 原版（全 classic, GPT-2 配方）| 1.7712 | 0.81 M |
| 完全體（全 modern, LLaMA 配方）| **1.5691** | **0.73 M** |

**−0.202（低 11%），而且參數更少。** 關鍵：改善幾乎「相加」（天真和預測 1.561、
實際 1.569）→ 四零件互補、各修不同地方（norm / MLP / 位置 / KV），所以能疊加。

## 兩個關鍵實驗

- **RoPE 外推**（`scripts/rope_extrapolation.py`）：兩模型都只用長度 64 訓練，
  測到 256——學習式位置 loss 從 1.93 爆到 2.79，RoPE 只從 1.82 緩升到 2.11。
  這就是長 context LLM 能「用比訓練更長序列」的關鍵。
- **GQA 權衡**：MHA 1.7712 / GQA 1.7974 / MQA 1.8126，KV-cache 1024/512/256。
  GQA 是甜蜜點——品質微降換 KV-cache 砍半，故 LLaMA-2/3、Mistral 都用它。

## 帶走的觀念（比數字更值錢）

1. **改進分兩口味**：準（降 loss）vs 省（省算力/記憶體）。評估新技巧前先問它優化哪個。
2. **「省」的零件 loss 不降才是成功**（RMSNorm 同準更省、GQA 換 KV-cache）。別預期每個現代技巧都降 loss。
3. **預測會錯**（RMSNorm 人猜錯口味、RoPE 連 Claude 也猜錯）→ 靠實測不靠嘴。
4. **互補零件好處會疊加** → 現代 LLM 是「全用」不是「挑一個」。
5. **工程紀律**：每個零件 = config 開關 + 單元測試 + 進監控面板 + commit 前 `make test` gate。

## 怎麼跑

```bash
# 單獨開某個零件
python pipeline/02_train.py --run_name rope --use_rope
# 完全體
python pipeline/02_train.py --run_name modern --use_rmsnorm --use_swiglu --use_rope --n_kv_head 2
# RoPE 外推 demo
python scripts/rope_extrapolation.py
```

## See Also

- [[theory-map]] — 哪些在聖經本、哪些是論文
- `TODO.md` — 架構已現代化，剩下的問題在別的維度（規模 / 後訓練 / 效率 / 嚴謹 / 工程）
- `README.md` 關鍵數據第 6 節
