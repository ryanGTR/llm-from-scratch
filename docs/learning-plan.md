---
title: 學習清單 — 即時深挖（邊做邊補）
type: plan
created: 2026-06-19
updated: 2026-06-19
tags: [learning, plan, just-in-time, ryan]
sources: [docs/theory-map.md, README.md]
---

# 學習清單（即時深挖）

給 Ryan 的個人化學習地圖。原則：**不重讀四年理論，而是「會反覆遇到的深挖、
前沿的先知道」**，而且每一項都綁一個「邊做邊補」的觸發點——做專案的某步時
順便把它補起來。背景：Java 工程師、目標偏應用工程（非研究），聖經本在手邊。

## 深度分三級

- 🔴 **深挖**：每次訓練都會撞到，debug 一定要真懂。值得花時間。
- 🟡 **夠用**：懂 what/why、能用、能推理，但不必會數學推導。
- ⚪ **知道即可**：知道它存在、在解什麼問題；真要用再深挖。

## 第一層：地基（🔴 深挖，聖經本 Ch5-8 就在手邊）

這些一直回來，是 debug 的本錢。撞到就翻書深挖。

| 主題 | 深度 | 邊做邊補的觸發點 | 書/出處 |
|---|---|---|---|
| loss 是什麼 / cross-entropy / 最大概似 | 🔴 | 看 `02_train.py` 的 loss、問「為什麼初始 loss≈ln(vocab)」 | Ch6 |
| 過擬合 / train-val gap | 🔴 | 看 `make plot-loss` 曲線，train 遠低於 val 時 | Ch5,7 |
| 最佳化：學習率 / Adam / 梯度裁剪 | 🔴 | 調 lr 跑壞一次（loss 爆或不動），再修 | Ch8 |
| 反向傳播的直覺 | 🔴 | 不用手推，但要懂 `loss.backward()` 在算什麼 | Ch6.5 |
| 正則化：dropout / weight decay | 🟡 | 改 `config.py` 的 dropout 看 val 變化 | Ch7 |
| 權重初始化為何重要 | 🟡 | 把 init 改壞（std=1.0）看 loss 爆掉 | Ch8.4 |

## 第二層：架構（🔴🟡 這是你現在的主戰場，論文不是書）

| 主題 | 深度 | 邊做邊補的觸發點 | 出處 |
|---|---|---|---|
| self-attention（Q/K/V、softmax、加權） | 🔴 | 已講＋熱圖；換 layer/head 多看幾張 | Transformer 2017 |
| 因果遮罩 / context window | 🔴 | 已測 block_size 上限；做 128 vs 256 實驗 | Transformer / GPT |
| multi-head 為何要多頭 | 🟡 | 畫不同 head 的熱圖，找出分工差異 | Transformer 2017 |
| tokenization：char vs BPE（subword） | 🔴 | 學習弧線的 C：換 BPE，比較 vocab/壓縮比 | Sennrich 2016 |
| position embedding | 🟡 | 把 pos_emb 拿掉看生成壞掉 | Transformer / GPT |
| LayerNorm / residual / GELU | 🟡 | 拿掉 residual 看深層訓不動 | 各論文 2015-16 |

## 第三層：系統 / 規模（⚪ 知道即可，要用再挖）

| 主題 | 深度 | 何時才深挖 | 出處 |
|---|---|---|---|
| FlashAttention（O(T) 記憶體） | ⚪ | 已知道在解什麼；真要訓長 context 再挖 | Dao 2022 |
| KV-cache（推論加速） | ⚪ | 之後做「快速生成 / 部署」時 | GPT 推論 |
| 量化 / 混合精度 | ⚪ | 顯存不夠、要塞大模型時 | — |
| 分散式 / 多卡訓練 | ⚪ | 上雲、單卡裝不下時 | — |
| LoRA / 微調 | 🟡 | 學習弧線之後的 B 路線（拿開源權重微調）| Hu 2021 |

## 把「看過」轉成「會」的練習法

每個概念都套這四個動作（這才是內化，不是再讀一遍）：

1. **改** — 動一個參數/一行 code
2. **預測** — 跑之前先猜結果會怎樣
3. **弄壞** — 故意改錯，看它怎麼壞（壞法會教你它在幹嘛）
4. **講解** — 試著用一句話講給別人聽，講不清就是還沒懂

> 能在跑之前「預測對」、壞掉能「說出為什麼」、能「講給別人聽」——這三個達到了，就是真懂了。

## 進度勾選

- [ ] 🔴 cross-entropy / 為什麼初始 loss≈ln(vocab)
- [ ] 🔴 過擬合（從 loss 曲線讀出來）
- [ ] 🔴 學習率調壞再修
- [ ] 🔴 context window：128 vs 256 連貫度實驗
- [ ] 🔴 BPE tokenizer（換掉 char-level）
- [ ] 🟡 multi-head 分工（多看熱圖）
- [ ] 🟡 拿掉 residual 看深層訓不動

## See Also

- [[theory-map]] — 程式碼 ↔ 章節/論文
- `README.md` — 專案 roadmap
