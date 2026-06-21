# llm-from-scratch

![CI](https://github.com/ryanGTR/llm-from-scratch/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-cu128-ee4c2c.svg)

從零手刻一個小型 GPT，並把「資料 → 訓練 → 評估 → 生成」做成一條可重跑、
可驗收、可監控的 pipeline。目標是**搞懂 LLM 原理**，同時練習**模型生產流水線的
工程化**。從本機（Framework 16 + RTX 5070）起步，結構乾淨到可平滑搬上雲端 GPU。

> 這不是要做 ChatGPT。這裡的模型是 ~0.1–1M 參數的 GPT，小到能在自己機器上
> 幾十秒訓完，但麻雀雖小五臟俱全——self-attention、multi-head、residual、
> causal mask、自回歸生成、BPE tokenizer、訓練迴圈，全都是真的、可跑、可改。

## 這個專案涵蓋什麼

- **資料工程**：collect → clean → 去重（MinHash）→ tokenize → pack，附驗收 playbook 與品質指標
- **模型**：decoder-only Transformer（`src/model.py`），跟 GPT-2/3 同一張架構藍圖，
  並可一鍵切換現代零件（RMSNorm / SwiGLU / RoPE，LLaMA/Mistral 同款）
- **訓練/評估/生成**：完整自回歸 pipeline，GPU 上跑
- **監控**：loss 曲線、過擬合偵測、attention 熱圖、tokenizer 對比，全部在 Jupyter 統一面板
- **tokenizer**：char-level 與自刻 BPE 兩種，可一鍵切換對比
- **可驗證**：18 個單元測試 + 驗收 playbook（`make verify`）

## 關鍵數據與發現

全部在 RTX 5070 Laptop（8.1 GB VRAM）+ torch 2.11.0+cu128 上實測，語料為
tiny-shakespeare（1,115,394 字元）。每個發現都是「先預測、再實測、看曲線」得來的。

### 1. 第一個從零訓練的模型

| 項目 | 數值 |
|---|---|
| 參數量 | 0.81 M |
| 訓練資料 | 1M token（char-level）|
| 訓練步數 / 時間 | 3000 步 / **35 秒（GPU）** |
| val loss | 4.20 → **1.77**（亂猜基準 ln(65)=4.17）|

生成（prompt `ROMEO:`）已長出莎士比亞的「形狀」：劇本對白格式、真實英文字的
拼寫節奏——它從 1 MB 文字、35 秒，自己學會了英文正字法與劇本結構。

### 2. Context window：放大不一定有用（容量受限）

block_size 128 vs 256，其餘相同：

| block_size | best val loss |
|---|---|
| 128 | 1.771 |
| 256 | 1.745（≈ 平手）|

**發現**：把記憶窗加倍只換來微幅進步。因為 0.81M 的小模型「用不到」更長的
context——長 context 要有相應的模型容量與資料才吃得到。這也是真實 LLM
「參數、資料、context 一起放大」的原因。

### 3. Residual connection 是「承重牆」不是裝飾

拿掉 `model.py` 每個 block 的 `x +`（殘差連接）重訓：

| 設定 | best val loss |
|---|---|
| 有 residual | **1.77**（順利下降）|
| 無 residual | 3.35（**卡死**，≈ 亂猜）|

**發現**：少一個加法，4 層模型就因梯度消失而幾乎學不動。residual 給梯度一條
「原封不動傳回前層」的高速公路，是能訓練深層網路的關鍵（ResNet 2015）。

### 4. Tokenizer：char vs BPE，以及「別被 raw loss 騙」

同 config 各訓一個（BPE 用 300 merges）：

| tokenizer | vocab | token 總數 | 每 token 字元 | val loss | **BPC（bits/char）** |
|---|---|---|---|---|---|
| char | 65 | 1.12 M | 1.00 | 1.77 | 2.56 |
| bpe | 365 | 0.55 M | 2.03 | 3.34 | **2.37** |

**關鍵發現**：BPE 的 raw val loss（3.34）看起來比 char（1.77）差很多——但這是
**陷阱**。兩者 vocab 不同（從 65 類 vs 365 類裡猜），raw loss 不可直接比。換成
tokenizer 無關的 **BPC（每字元幾 bit）**，BPE 反而**更好**（2.37 < 2.56），而且
只用**一半的序列長度**。教訓：跨 tokenizer 比較一定要用 BPC，別被表面數字騙。

### 5. 這台機器能跑多大 context（naive vs FlashAttention）

| attention 實作 | 最大 block_size（batch 32, 8GB）|
|---|---|
| 樸素（攤開 T×T，O(T²)）| ~1024 |
| FlashAttention（O(T)）| ~4096 |

**發現**：同一台機器、只換 attention 算法，context 上限 4 倍。樸素 attention
記憶體隨 context 平方成長，FlashAttention 改成線性——這就是真實 LLM 達到
128k context 的關鍵，不是靠顯存大 1000 倍。

### 6. 現代架構零件（LLaMA/Mistral 同款）升級

把 2017 原版 Transformer 零件換成 2023+ 主流（皆可用 config 開關切換對比）：

| 零件升級 | best val loss | 改善 | 口味 |
|---|---|---|---|
| LayerNorm → **RMSNorm** | 1.7712 → 1.7724 | ≈ 平手 | 更省（少計算）|
| GELU MLP → **SwiGLU** | 1.7712 → 1.6810 | **−0.09** | 更準（gating 表達力）|
| 學習式位置 → **RoPE** | 1.7712 → 1.6248 | **−0.146**（且參數更少）| 更準 + 外推 |

**發現一**：不是每個「現代技巧」都降 loss——RMSNorm 優化的是「成本」（同準、更省），
SwiGLU / RoPE 優化的是「準度」。評估新技巧前要先問「它優化的是準還是省」。

**發現二（RoPE 外推）**：兩個模型都只用長度 64 訓練，測試餵到 256——學習式位置
loss 從 1.93 爆到 2.79（沒訓過的位置就垮），RoPE 只從 1.82 緩升到 2.11（優雅外推）。
這就是長 context LLM 能「用比訓練更長的序列」的關鍵。見 `scripts/rope_extrapolation.py`。

**嚴謹度（多 seed 驗證）**：各跑 3 個 seed 取 mean ± std（`scripts/multi_seed.py`）——
classic 1.787±.007 / SwiGLU 1.678±.010 / RoPE 1.625±.006。SwiGLU、RoPE 的改善
是雜訊的 11–22 倍 → **確認為真差異**；雜訊地板 ≈ 0.01，故 RMSNorm（0.001）、
128↔256（0.026）屬雜訊等級 = 平手。教訓：單一裸數字會騙人（單次 classic 1.7712
其實是偏低的一抽，真實 mean 1.787），報數據要附 ±std。

### 7. 中文主線：真實規模 + 真實語言（資料工程閉環）

從「1MB 英文玩具」放大到「105MB 真實中文維基」（11,126 篇、3,990 萬字），讓資料
pipeline 在真實規模 + 一個新語言上實戰。一路撞破三個「英文／小資料假設」並修復：

| 撞破的假設 | 症狀 | 修法 |
|---|---|---|
| 近似去重用「空白切詞」做 shingle | 中文沒空白 → 去重失效 | 改**字元 n-gram**（語言無關）|
| 兩兩比對 O(n²) | 11k 篇純 Python 跑不動 | 加 **MinHash + LSH** → 近 O(n)，21.6 秒去完、抓 161 篇近似重複 |
| 字元熵門檻（3.5–6）是英文校準 | 中文 9.70 bits 被誤判❌ | 改**熵效率＝熵/log2(vocab)**（英 0.78／中 0.70 都健康）|

**資料品質報表系統**（`src/data/quality_report.py`，`make quality`）：把「看資料」工業化成
偵測器（wiki 語法／HTML／控制字元／URL／符號／重複／過短），掃全語料、量化、設門檻 = 資料
gate。第一跑就抓到**聚合指標漏掉的問題**——維基 `-{zh-tw:..;zh-cn:..}-` 繁簡轉換語法漏過
清洗、**21.63%（2,407 篇）中招**。修 `clean.py` 後 **21.63% → 0.05%、gate 由 ❌ 轉 ✅**
（監控面板有 before/after 對照圖）。教訓：熵／壓縮等聚合指標必要但**遠遠不夠，一定要看樣本**。

**訓練成果**（6 層 / 256 維 / 現代架構全開，4000 步）——訓練前先講好成功判準，逐項驗證：

| 成功判準 | 結果 | 達標 |
|---|---|---|
| 有沒有在學（loss vs 亂猜基準 ln14210=9.56）| 9.59 → val **3.67** | ✅ |
| 會不會類推（test ≈ val？過擬合？）| test 3.695 ≈ val 3.677（gap 0.018）| ✅ |
| 學了多少（BPC vs 無條件熵 9.70）| **5.33** bits/char（靠 context 砍 45%）| ✅ |
| 像不像中文（質性）| 真詞／語法／標點對，全域不連貫=小模型水準 | ✅ |

生成樣本（餵「數學是」）：「數學是一種特殊的理論，如經濟學和哲學的一些特征…例如，這個
基因分子的第二個基因分子序子的特徵分子…」——真詞、學術語域、明顯比英文亂碼有感。

**主線一句話**：抓資料 → 改去重 → 品質報表 → 接面板 → 修清洗（前後對照）→ 重備 → 訓練 →
四判準評估。每步「先講評估方式、再做、如實對結論」= 真實 MLOps 資料品質迴圈。

## 心智模型（Java 類比）

| 這個專案 | Java 世界 |
|---|---|
| `src/model.py` 前向/訓練邏輯 | 你的 business logic |
| `Makefile` 串各階段 | Spring Batch job：step1→step2→… |
| `artifacts/ckpt.pt` 權重檔 | 編譯產物 `.jar` |
| tokenizer（char / BPE）| serializer，String ↔ int[] |
| self-attention | 模糊版 `HashMap.get()`：對所有 key 算相似度、回傳加權平均 |
| `make verify` / `tests/` | JUnit + 驗收測試 |
| `uv` | SDKMAN!（管 Python 版本）+ Maven（管依賴）合體 |

## 結構

```
llm-from-scratch/
├── src/
│   ├── config.py          # 所有超參數（= application.yml）
│   ├── tokenizer.py       # char tokenizer + load_tokenizer() 自動辨識
│   ├── bpe.py             # 自刻 BPE：train_bpe + BPETokenizer
│   ├── model.py           # GPT + 可切換現代零件（RMSNorm/SwiGLU/RoPE）
│   ├── viz.py             # 用 forward hook 抓 attention 權重
│   ├── data/              # 資料子系統（純 Python、零依賴）
│   │   ├── sources.py     #   collect：來源 → Document
│   │   ├── clean.py       #   normalize + 品質過濾
│   │   ├── dedup.py       #   exact + near-dup（MinHash）
│   │   └── stats.py       #   資料品質指標（量/熵/壓縮比/重複率）
├── pipeline/
│   ├── 01_prepare_data.py # collect→clean→dedup→tokenize→pack（--tokenizer char/bpe）
│   ├── 02_train.py        # 訓練（loss 記到 runs/，自動偵測 GPU）
│   ├── 03_eval.py         # val loss / perplexity
│   ├── 04_generate.py     # 自回歸生成
│   ├── plot_loss.py       # loss 曲線（多 run 疊圖）
│   └── viz_attention.py   # attention 熱圖（單張 / --grid 全部 head）
├── scripts/
│   ├── get_data.sh        # 下載樣本語料
│   ├── make_messy_corpus.py # 產髒語料示範清洗/去重
│   ├── train_bpe.py       # 跑 BPE + 可審核監控（合併 log / report）
│   ├── rope_extrapolation.py # RoPE 外推 demo（train@64, eval→256）
│   ├── multi_seed.py      # B④ 多 seed 嚴謹：mean±std + 誤差線 + 真差異判定
│   └── verify.py          # 驗收 playbook 執行器
├── notebooks/
│   ├── 01_explore_data.ipynb # 資料探索
│   └── 02_monitor.ipynb      # 統一監控面板（BPE / 訓練 / 資料品質）
├── tests/test_data.py     # 18 個單元測試
├── docs/                  # 延伸文件（見 See Also）
├── Makefile               # pipeline orchestrator（make help 看全部）
└── matplotlibrc           # 讓圖表中文正常
```

## 環境

系統 Python 是 3.14（PyTorch 還沒支援），所以用 **uv** 釘一個獨立的
**Python 3.12** 環境。CUDA 運算不需要 `prime-run`（那只用於顯示 offload）。

```bash
cd ~/Documents/llm-from-scratch
# 方式 A（可重現，推薦）：依 uv.lock 裝「鎖死的精確版本」
uv sync                          # 讀 pyproject.toml + uv.lock，建好一致的環境

# 方式 B（手動）：
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy matplotlib jupyterlab pandas tiktoken
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> 依賴版本鎖在 `pyproject.toml` + `uv.lock`（132 套件全部 pin 死）→ 換機器 `uv sync` 拿到
> 「位元級一致」的環境，這是 MLOps 可重現性的基本功。
> 資料 pipeline（`make test / verify / data-demo / stats / bpe`）只需 **numpy**（不需 torch）——
> 近似去重用 MinHash + LSH 向量化以擴展到上萬篇，故依賴 numpy。

## 跑跑看

```bash
make help          # 列出所有指令

# 資料（不需 torch）
make data-demo     # 產髒語料，看清洗/去重各砍多少
make stats         # 資料品質指標 + 健康判讀
make verify        # 驗收 playbook：逐項 PASS/FAIL
make bpe           # 跑 BPE，輸出可審核的合併 log

# 訓練 / 評估 / 生成（需 GPU 版 torch）
make smoke         # 極小設定快速跑通整條鏈
make data          # 下載莎士比亞 + tokenize（char）
make train         # 訓練（device 自動偵測 cuda）
make eval          # val loss / perplexity
make gen           # 讓模型續寫
make plot-loss     # loss 曲線（多 run 疊圖）
make attn          # attention 熱圖

# 換 BPE 重跑對比
python pipeline/01_prepare_data.py --tokenizer bpe --merges 300
python pipeline/02_train.py --run_name bpe_tok

# 現代架構零件（LLaMA 同款，可單獨或合併開）
python pipeline/02_train.py --run_name modern --use_rmsnorm --use_swiglu --use_rope
python scripts/rope_extrapolation.py   # RoPE 外推 demo（train@64, eval→256）

# 統一監控面板（含資料/BPE/訓練/tokenizer 對比/單元測試五大區塊）
make lab           # Jupyter → notebooks/02_monitor.ipynb → Restart & Run All
```

## 學習弧線（本專案的設計脈絡）

B 監控 → D 原理 → A 調參 → C tokenizer，口訣「B 給眼睛、D 給腦、A 動手、C 升級」：

- **B 監控**：loss 曲線、過擬合 gap、統一監控面板（先有儀表才好做實驗）
- **D 原理**：self-attention（Q/K/V/熱圖）、causal mask、multi-head、residual
- **A 調參**：context window、模型大小，邊改邊看曲線（見上方關鍵數據）
- **C 升級**：char → BPE subword tokenizer，並學會用 BPC 公平比較

## See Also

- `docs/theory-map.md` — 程式碼 ↔《Deep Learning》聖經本章節 / 論文 對照地圖
- `docs/learning-plan.md` — 即時深挖學習清單（哪些深挖、哪些知道即可）
- `docs/data-pipeline.md` — 資料這塊各階段在做什麼
- `docs/verification-playbook.md` — 驗收 runbook
- Karpathy, *Let's build GPT: from scratch* / **nanoGPT**（本專案精神來源）
- Vaswani et al. 2017, *Attention Is All You Need*；Sennrich et al. 2016（BPE）
