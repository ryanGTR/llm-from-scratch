---
title: 理論地圖 — 程式碼 ↔ 聖經本章節 / 論文
type: map
created: 2026-06-19
updated: 2026-06-21
tags: [llm, transformer, gpt, deep-learning-book, papers, learning-map]
sources:
  - src/model.py
  - src/reward_model.py
  - pipeline/02_train.py
  - pipeline/06_dpo.py
  - pipeline/08_grpo.py
  - "Goodfellow, Bengio, Courville — Deep Learning (2016)"
  - "Vaswani et al. — Attention Is All You Need (2017)"
---

# 理論地圖

把本專案手刻的每一塊，釘到它出自哪裡：**聖經本《Deep Learning》(Goodfellow,
Bengio, Courville, 2016)** 的章節，或 **後續論文**。用途：回頭看時知道「我學的
這段在學界地圖的哪裡、該去翻哪本/哪篇」。

## 時間軸（為什麼有些東西聖經本沒有）

```
2015  ResNet（residual）
2016  ← 聖經本出版。LayerNorm、GELU、BPE 也都這年的論文
2017  Transformer（Attention Is All You Need）；PPO；RLHF from preferences（Christiano）
2018  GPT-1 / 2019 GPT-2（decoder-only 自回歸 LM）；RMSNorm 2019；nucleus sampling 2019
2019  Fine-tuning LM from human feedback（KL 錨用到 LM）
2020  SwiGLU；2021 RoPE
2022  FlashAttention（O(T) attention）；InstructGPT（RLHF 配方）
2023  GQA；DPO；reward model 過度優化
2024  GRPO（DeepSeekMath）；min-p sampling → 2025 DeepSeek-R1
```

聖經本是 2016 年的「地基教科書」。本專案的「訓練機器」幾乎都在書裡，「現代架構 / 效率 /
取樣 / 後訓練對齊」幾乎都在書之後的論文（2017–2025）——下面三層逐塊釘到出處。

## 第一層：在聖經本裡（訓練地基）

本專案的整支訓練迴圈 `pipeline/02_train.py` 幾乎 100% 是這層。

| 程式碼 | 位置 | 聖經本章節 |
|---|---|---|
| 前饋網路 MLP | `src/model.py:56` (MLP class) | Ch6 Deep Feedforward Networks |
| softmax 輸出 + cross-entropy loss | `src/model.py:124` (F.cross_entropy) | Ch6（輸出單元 / 最大概似估計）|
| 反向傳播 | `pipeline/02_train.py:93` (loss.backward) | Ch6.5 Back-Propagation |
| 權重初始化 (std=0.02 常態) | `src/model.py:100` (_init_weights) | Ch8.4（初始化策略）|
| Dropout | `src/model.py:63,92` | Ch7.12 Regularization |
| Weight decay (L2) | `pipeline/02_train.py:72` (AdamW) | Ch7.1 |
| 過擬合 / train-val 切分 | loss 曲線的 train↔val gap | Ch5.2-5.3（容量、過擬合）|
| 最佳化器 Adam | `pipeline/02_train.py:72` | Ch8.5（Adam）|
| 梯度裁剪 | `pipeline/02_train.py:94` (clip_grad_norm) | Ch10.11.1 |
| 詞嵌入 embedding 的概念 | `src/model.py:90` (token_emb) | Ch12.4（NLP）|
| 「預測下一個 token」語言模型框架 | `src/model.py:111` (forward) | Ch12.4（n-gram / neural LM）|

## 第二層：聖經本「之後」的論文（架構 + 系統）

書裡沒有，是本專案真正「新」的部分——你現在站的位置。

| 程式碼 | 位置 | 出處論文 |
|---|---|---|
| self-attention / multi-head / 因果遮罩 | `src/model.py:47-51` | Attention Is All You Need (Vaswani 2017) |
| decoder-only、自回歸生成 | `src/model.py:130` (generate) | GPT-1 (Radford 2018) / GPT-2 (2019) |
| context window 裁切 (`idx[:, -block_size:]`) | `src/model.py` generate 內 | GPT 系列（自回歸推論）|
| 學習式 position embedding | `src/model.py:91` (pos_emb) | Transformer / GPT |
| LayerNorm | `src/model.py:74,76` | Ba et al. 2016（書裡是 BatchNorm, Ch8.7）|
| GELU 激活 | `src/model.py:66` (F.gelu) | Hendrycks & Gimpel 2016（書裡用 ReLU）|
| Residual connection | `src/model.py:80-81` | ResNet (He et al. 2015)（書只略提）|
| Weight tying | `src/model.py:97` | Press & Wolf 2017 / Inan 2016 |
| BPE subword tokenizer | `src/bpe.py`（train_bpe / BPETokenizer）✅ | Sennrich et al. 2016 |
| MinHash + LSH 近似去重 | `src/data/dedup.py` | Broder 1997（MinHash）；Indyk & Motwani 1998（LSH）|

### 現代零件（LLaMA / Mistral 同款，路線 A 全配齊 ✅）

書沒有、2017 Transformer 也沒有；這是把骨架升級成 2023+ 主流的那批。

| 程式碼 | 位置 | 出處論文 |
|---|---|---|
| RMSNorm | `src/model.py` (RMSNorm class, `use_rmsnorm`) | Zhang & Sennrich 2019 |
| SwiGLU（gated MLP）| `src/model.py` (SwiGLU class, `use_swiglu`) | Shazeer 2020「GLU Variants」|
| RoPE（旋轉位置編碼）| `src/model.py` (build_rope_cache/apply_rope, `use_rope`) | Su et al. 2021「RoFormer」|
| GQA / MQA（共用 KV 頭）| `src/model.py` (`n_kv_head`) | Ainslie et al. 2023（GQA）；Shazeer 2019（MQA）|

### 效率與取樣 ✅

| 程式碼 | 位置 | 出處論文 |
|---|---|---|
| FlashAttention（O(T) 記憶體）| `src/model.py` (`use_flash`, F.SDPA) | Tri Dao 2022 |
| KV-cache（增量生成）| `src/model.py` (cache_k/v, `use_kv_cache`) | 標準推論優化（無單篇定論；見 nanoGPT/推論系統）|
| top-p / nucleus 取樣 | `src/model.py` (_sample, `top_p`) | Holtzman et al. 2019「Neural Text Degeneration」|
| min-p 取樣 | `src/model.py` (_sample, `min_p`) | Nguyen et al. 2024「Min-p Sampling」|
| Deep ensemble（機率平均）| `scripts/deep_ensemble.py` | Lakshminarayanan et al. 2017 |

## 第三層：後訓練（對齊）論文 — SFT → DPO → RLHF

把「接龍機器」對齊成助理這條線。**注意新舊**：RLHF 骨架與 KL 錨其實是 2017–2022 的成熟
技術，DPO 是 2023，只有 **GRPO 是 2024–25 真正新的**——別把「對齊」整包當成最新研究。

| 程式碼 | 位置 | 出處論文（年份）|
|---|---|---|
| SFT（指令微調）| `pipeline/05_sft.py` | InstructGPT 的第一步 (Ouyang 2022)；概念更早 |
| 偏好模型 Bradley-Terry | `src/reward_model.py` (bt_loss) | Bradley & Terry **1952**（純統計，超老）|
| Reward model（學來的評分器）| `src/reward_model.py` | Christiano et al. **2017**；Ziegler et al. 2019 |
| RLHF + **KL-to-reference 懲罰** | `pipeline/08_grpo.py` (β·KL) | Ziegler **2019**（KL 錨用到 LM）；Ouyang 2022 |
| PPO（RL 本體，我們用 GRPO 替代）| —（對照）| Schulman et al. 2017 |
| **DPO**（免 RM、免 RL 的封閉式）| `pipeline/06_dpo.py` | Rafailov et al. **2023**（NeurIPS）|
| **GRPO**（去掉 critic，組內 baseline）| `pipeline/08_grpo.py` | DeepSeekMath (Shao **2024**)；DeepSeek-R1 (2025) |
| reward hacking / 過度優化 | `pipeline/08_grpo.py` 對照、`scripts/eval_grpo.py` | Gao et al. **2023**「RM Overoptimization」；Goodhart 1975 |
| IPO（治 DPO 過度優化，下一步選項）| 尚未做 | Azar et al. 2023/24 |

## 一句話：我們在哪

> 從 **聖經本 Ch5-8 訓練地基** 起步，手刻 **2017 Transformer + 2018 GPT 骨架**，再把它
> 升級成 **2019–2023 現代零件（RMSNorm/SwiGLU/RoPE/GQA、LLaMA 同款）+ 效率/取樣
> （FlashAttention/KV-cache/top-p/min-p）**，最後走完 **2022–2025 後訓練對齊（SFT→DPO→
> RLHF/GRPO）**。

學習路徑＝聖經本 Ch5-8 地基 → 2017 Transformer（`src/model.py`）→ 2018 GPT → 2019–2023
現代化零件 → 2022 效率工程 → **2023–2025 後訓練前沿（站在 DPO/GRPO 這條最新線上）**。

## 延伸閱讀

- **架構**：Vaswani 2017 "Attention Is All You Need"；Jay Alammar "The Illustrated
  Transformer"（圖解，最好入門）；Karpathy "Let's build GPT"（本專案精神來源）
- **現代零件**：Zhang & Sennrich 2019（RMSNorm）；Shazeer 2020（SwiGLU）；Su 2021（RoPE）；
  Ainslie 2023（GQA）——合起來就是 LLaMA/Mistral 的配方
- **取樣**：Holtzman 2019（nucleus/top-p）；Nguyen 2024（min-p）
- **系統**：Dao 2022 "FlashAttention"；Karpathy nanoGPT repo
- **後訓練/對齊**：Ziegler 2019（KL 錨）；Ouyang 2022（InstructGPT/RLHF 配方）；
  Rafailov 2023（DPO）；Shao 2024（DeepSeekMath/GRPO）+ DeepSeek-R1 2025；
  Gao 2023（reward model 過度優化＝reward hacking）
- **地基**：Goodfellow et al. 2016《Deep Learning》Ch5-8, Ch12

## See Also

- [[derivations]] — 關鍵演算法的**數學推導/證明**（本頁給「出處」，那頁給「為什麼成立」）
- `README.md` — 專案總覽與 roadmap
- [[data-pipeline]] — 資料這塊在做什麼
- [[verification-playbook]] — 怎麼驗收
- `src/model.py` — 上表大部分引用的所在
