# llm-from-scratch

![CI](https://github.com/ryanGTR/llm-from-scratch/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-cu128-ee4c2c.svg)

> 📖 **線上閱讀本專案的書**（Quarto，O'Reilly 風，含程式碼與數學）：
> **<https://ryangtr.github.io/llm-from-scratch/>**

從零手刻一個小型 GPT，並把它一路推完整個 **LLM ＋ MLOps 生命週期**：
**資料工程 → 現代架構 → 訓練評估 → 部署/治理 → 後訓練對齊（SFT→DPO/IPO→RLHF: GRPO/PPO）**，
全做成可重跑、可驗收、可監控的 pipeline。目標是**真正搞懂原理**（含關鍵演算法的數學推導），
同時用企業 IT 的治理視角走完一遍**真實 MLOps**。從本機（Framework 16 + RTX 5070）起步、可搬上雲。

> 這不是要做 ChatGPT。模型小（demo ~0.1M、實戰中文模型 ~8M、char-level），小到自己機器幾十秒～幾分鐘
> 訓完——但**麻雀雖小五臟俱全**：self-attention、RMSNorm/SwiGLU/RoPE/GQA（LLaMA 同款）、FlashAttention、
> KV-cache、BPE、reward model、GRPO/PPO/DPO/IPO，全都是親手刻的、可跑、可改、有對照實驗。

**最有代表性的幾個發現**（每個都「先預測 → 實測 → 被打臉 → 想懂為什麼」）：

- 🎯 **reward hacking 活體**：RLHF 拿掉 KL 錨，RM 分數衝 3.7→13.2，輸出卻 collapse 成「不管問什麼都吐同一串垃圾」（發現 10）
- 🧠 **死背 vs 真學**：DPO 兩種偏好軸 train-acc 都 100%，held-out 一個 97% 一個 9%——train 會騙你（發現 9）
- 🔬 **聚合指標的盲點**：熵/壓縮全說「資料健康」，偵測器卻抓到 21.6% 文件殘留維基語法（發現 7）
- ⚙️ **沒有「一定更快」**：KV-cache 在 GPU+小模型+短生成反而慢 18%——省的技巧要量你的 workload（發現 8）

> 📖 給人讀的完整旅程敘事（作品集入口）：[docs/case-study.md](docs/case-study.md)　·　📐 數學推導：[docs/derivations.md](docs/derivations.md)　·　📊 讀圖指南：[docs/reading-the-charts.md](docs/reading-the-charts.md)

## 拿這個 repo，你可以做什麼（依投入程度）

不只是「讀」——這 repo 的東西**按得下去、跑得出來**。依你的時間與設備選一條：

### 🟢 10 分鐘・零 GPU・任何筆電 ← 最推薦的起點

`book/examples/tiny_*.py` 是書裡每個結論的**自包含、純 CPU、幾十秒~幾分鐘**可重現版（只要 `torch`，
連 repo 其他東西都不用懂；用 1 MB 莎士比亞，不必抓 105 MB 中文）。每支都被 **CI 顧著、不會壞**：

| 跑這支 | 親手看到 |
|---|---|
| `tiny_gpt.py` | 從零訓一個會「假裝莎士比亞」的小 GPT（亂碼→有劇本形狀）|
| `tiny_modern.py` | 一鍵切 RMSNorm/SwiGLU/RoPE，驗證「準 vs 省」|
| `tiny_kvcache.py` | KV-cache 與樸素版**逐 token 對拍相同**、再看加速隨長度張開 |
| `tiny_eval.py` | 亂猜基準、BPC、多 seed mean±std（評估三紀律）|
| `tiny_dedup.py` | 聚合指標說「健康」但偵測器抓出 20% 髒資料；MinHash/LSH 候選暴減 |
| `tiny_observability.py` | 冷啟動 vs 暖機、p50/p95 尾延遲 |
| `tiny_serve.py` | digest 身份（認內容不認檔名）+ promotion gate 真的擋下 |
| `tiny_drift.py` | PSI 把資料漂移量化成一個數字、觸發重訓 |
| `tiny_dpo.py` | DPO「train-acc 衝 100%、held-out 才說真話」|

```bash
cd book/examples
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python tiny_gpt.py          # 或任一支；make book-smoke 會極小設定跑通全部
```

### 🟡 完整版・要 GPU + 環境（uv / torch-cu128 / podman）

`uv sync` 後跑完整 pipeline：訓真的 8M 中文模型、起 FastAPI 服務、Prometheus+Grafana 儀表板、
模型 registry + promotion gate、drift/重訓/金絲雀迴圈。入口在下面的「跑跑看」與線上書第三部。

### 🔵 借 pattern 進你自己的專案

把這些當「怎麼把治理接到 ML」的範式庫直接搬：`src/registry.py`（digest 身份 + gate）、
`src/data/quality_report.py`（資料品質偵測器）、`pipeline/06_dpo.py`／`08_grpo.py`（DPO/RLHF 損失）。

> 誠實邊界：這是**學習 / 作品集** repo，不是維護中的 library；目標是「真正搞懂 + 可重現地證明」，
> 不是做出 ChatGPT（模型是 8M 玩具尺度）。

## 這個專案涵蓋什麼

- **資料工程**：collect → clean → 去重 → tokenize → pack，附驗收 playbook、品質指標與**偵測器報表**
- **真實規模資料**：105 MB 中文維基實戰——字元 n-gram + **MinHash LSH** 去重（O(n²)→近 O(n)）、
  資料品質報表抓出聚合指標漏掉的問題（見發現 7）
- **模型**：decoder-only Transformer（`src/model.py`），跟 GPT-2/3 同一張架構藍圖，
  並可一鍵切換現代零件（RMSNorm / SwiGLU / RoPE / GQA / FlashAttention / KV-cache，LLaMA/Mistral 同款）
- **訓練/評估/生成**：完整自回歸 pipeline，GPU 上跑，附「先講判準再驗證」的嚴謹評估
- **部署與治理（MLOps）**：FastAPI 推論服務、Prometheus + Grafana 可觀測性、Podman GPU 容器化、
  模型 registry + lineage + model card + promotion gate（見發現 8）
- **監控**：loss 曲線、過擬合偵測、attention 熱圖、資料品質 before/after，全部在 Jupyter 統一面板
- **tokenizer**：char-level 與自刻 BPE 兩種，可一鍵切換對比
- **可驗證**：單元測試 + 驗收 playbook（`make verify`）+ CI（GitHub Actions）
- **後訓練對齊**：SFT → DPO → RLHF（reward model + GRPO），含 reward hacking 示範（見發現 9–10）

> 📚 **每塊程式碼出自哪篇論文？** 看 [`docs/theory-map.md`](docs/theory-map.md) 的三層對照地圖
> （訓練地基 → 現代架構/效率/取樣 → 後訓練對齊），每項都釘到出處論文與年份、並標明新舊。

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

### 8. 部署下半場：把模型變成可治理的線上服務（MLOps serving）

訓練是「離線、一次性、看 loss」；服務是「線上、持續、面對真實流量」——要快、要被呼叫、
要看得到、要可重現、要可治理。下半場把訓練好的模型一路推到「能上線、能稽核」。

| 階段 | 做了什麼 | 關鍵檔案 / 指令 |
|---|---|---|
| a. 推論 API | FastAPI 包模型：`/health`(就緒探針) `/generate`(回延遲+吞吐) `/model`(治理) | `serve/app.py`、`make serve` |
| b. 可觀測性 | Prometheus `/metrics`(請求數/延遲 histogram/token) + 結構化 JSON 日誌 | middleware 自動量每請求 |
| c. 容器化 | Podman + GPU（CDI passthrough）+ 模型 runtime mount（與映像解耦） | `Containerfile`、`make image`/`run-container` |
| d. 監控儀表板 | Prometheus + Grafana（podman pod，dashboard 自動 provisioning） | `monitoring/`、`make dashboard` |
| e. 模型治理 | digest 身份 + registry 台帳 + lineage + model card + promotion gate | `src/registry.py`、`make register`/`models` |

**API 端點**（`make serve` 起，base `http://127.0.0.1:8000`，注意是 127.0.0.1 不是 localhost）：

| 方法 | 路徑 | 用途 |
|---|---|---|
| `GET` | `/health` | 就緒探針：模型載好沒 / device / 參數量 / vocab / 批次統計 |
| `POST` | `/generate` | 生成（主端點）→ 回 text + latency + tokens/s + variant + served_digest |
| `GET` | `/drift` | e1 漂移：PSI / OOV / level / retrain_suggested |
| `GET` | `/model` | 治理：服務的 digest + registry 狀態 + 金絲雀資訊 |
| `GET` | `/metrics` | Prometheus 指標（給 Grafana 抓） |
| `GET` | `/docs` | Swagger 互動文件（FastAPI 自動產，可直接點測） |

```bash
# /generate body：欄位都有預設，最少只要 prompt
curl -s 127.0.0.1:8000/generate -H 'content-type: application/json' \
     -d '{"prompt":"數學是","max_new_tokens":80,"temperature":0.8,"top_p":0.9}'
```

**效能實測（服務第一課：量你的 workload，別假設）**：KV-cache 在 CPU 長生成快 2.2×，
但在「GPU + 小模型 + 短生成」反而慢 ~18%（一次算一個 token 浪費 GPU 平行度，per-step 開銷 >
省下的 O(T²) 重算）。「省」的技巧是否真省，看 regime。可觀測性也立刻抓到**冷啟動**：第一個
請求 ~354ms、暖機後 ~50ms（首次 CUDA kernel 編譯）。

**模型治理 = 對「線上的模型」答得出四個稽核問題**：
1. 哪一個？→ 模型身份用 **ckpt 的 sha256 digest**（像 container image digest / cosign，不靠檔名）
2. 吃什麼訓的？→ **lineage**：每筆綁 資料 digest + 資料品質 gate + config + git commit
3. 表現如何？→ **model card**（`registry/cards/<digest>.md`，人讀單據）
4. 憑什麼上線？→ **promotion gate**：沒過資料品質 gate + test 評估就「擋下」，上線是被 enforce 的
   。服務端 `/model` 回報自己的 digest + registry 狀態 → 確認線上是不是被批准那顆（`UNREGISTERED`=紅旗）。

> 多容器（API+Prometheus+Grafana）用 **podman pod** 一起跑、共享 localhost——正是 k8s「Pod」
> 概念的微縮版。單機這樣剛好；要多副本/跨機/自動擴縮才需要 k8s。

**進階（e）——治理線 + 效能線**：

| 項 | 做什麼 | 實測 / 開關 |
|---|---|---|
| e1 漂移監控 | 線上請求 vs 訓練分布（OOV + 分箱 PSI），偏離就建議重訓 | 中文 PSI 0.07 穩定 / 英文 3.83 漂移；`/drift` |
| e2 重訓迴圈 | 資料→訓練→評估→註冊→gate→promote 自動外圈 + **回歸 gate**（更爛的擋下） | `make retrain`；demo 較差模型被 gate 擋 |
| e3 金絲雀/A-B | 同時載兩顆模型、導 N% 流量到候選、按 variant 分標比較 | `CANDIDATE_CKPT` + `CANARY_PCT`；回滾=設 0 |
| e4 動態批次 | async 佇列把併發請求合批一次算，衝 GPU 吞吐 | `BATCH_MAX`；32 併發 **4.4×**（42 vs 9.5 req/s）|
| e5 模型壓縮 | fp16 / int8 量化，量「大小 vs 品質」取捨 | `make compress`；fp16 **2.0×** near-lossless |

> e1–e3 是「治理線」（偵測→重訓→漸進放量），e4–e5 是「效能/成本線」。權重量化（e5）跟
> KV-cache 量化（TurboQuant）是不同軸、可疊加。

**產出正確性（不只比效能）**：服務最容易只盯延遲/吞吐，忘了驗「答得對不對」。三道檢查：
- **批次不變量**：合批生成必須逐字等於單獨生成（`tests/` 有測；e4 只該更快、不該改輸出）。
- **模型比較**：`make compare A=… B=…` → 兩模型的 test_loss + greedy 一致率（離線）。
- **shadow agreement**：候選「在影子裡」跟著算同一輸入、回報與現行的 next-token 一致率（線上、
  `SHADOW_PCT`、不拖延遲）→ dashboard 面板。

**放量決策**：硬條件是「品質 gate」，dashboard 是「看變化幅度與副作用」——別只看單一數字：

| 看什麼 | 訊號 | 怎麼讀 |
|---|---|---|
| 品質（能不能上）| promotion gate：test_loss 不比現行差 | **這是 pass/fail 的硬條件** |
| 變化幅度 | ⑨ shadow agreement | 高=幾乎沒變；**低=改很多（不等於壞）**——配合品質看 |
| 沒變慢 | ⑦ canary p95 延遲 | canary ≈ production |
| 輸入正常 | ⑤ 漂移 PSI | < 0.25 |

> ⚠️ 重要 nuance（真實案例教的）：agreement 是「改了多少」不是「好不好」。8000 步候選
> test_loss 3.46（比現行 3.70 好），但 agreement 只有 **67.5%**——低一致率「不是壞事」，是它
> 真的學到更好的東西。**更好的模型本來就該跟舊的不一樣**。所以決策靠品質 gate（更好→放行），
> agreement 只告訴你「爆炸半徑多大、要多謹慎驗」。把 agreement 當硬門檻會擋掉每一次真進步。
> （合成候選那種 96% 高一致是「幾乎沒改」，反而才該問「那升它幹嘛」。）

```bash
cp <候選>.pt artifacts/candidate.pt        # 放一顆候選
BATCH_MAX=8 make dashboard                  # 自動開金絲雀+shadow，開 127.0.0.1:3000
# 品質 gate 過 + 延遲沒變糟 + 漂移正常 → 漸進加大 CANARY_PCT；有狀況 → 設 0 秒回滾
```

### 9. 後訓練：把「接龍機器」對齊成「會聽話、有偏好」（SFT → DPO）

預訓練只學「猜下一字」。後訓練分兩步把它對齊成助理：

| 步驟 | 學什麼 | 機制 | 關鍵檔案 |
|---|---|---|---|
| **SFT**（里程碑1）| 會聽話的**對話格式** | 在「問：…答：…」格式上續訓，行為從「續寫維基」變「應答」 | `pipeline/05_sft.py`、`make sft` |
| **DPO**（里程碑2）| 在兩個回答間**偏好較好的** | 一條封閉式損失直接優化偏好，不需 reward model（比 RLHF 簡單） | `pipeline/06_dpo.py`、`make dpo` |

**DPO 損失**（policy 從 SFT 起步、reference 是凍結的 SFT 當錨）：

```
loss = -log σ( β·[ (logπ(chosen) − logπ_ref(chosen)) − (logπ(rejected) − logπ_ref(rejected)) ] )
```

直覺：拉高「policy 相對 reference 對 chosen 的對數機率」、壓低 rejected；β 控制偏離 reference
的力度（錨防止為迎合偏好而崩壞）。單元測試守住數學：policy==reference 時 margin=0、loss=log 2。

**核心發現 ——「容量內就類推、超出就只會背」**（兩種偏好軸對照，`make eval-dpo` 出圖）：

| 偏好軸 | chosen vs rejected | held-out 偏好準確率 | 解讀 |
|---|---|---|---|
| **format**（易）| 連貫定義 vs 退化重複迴圈 | SFT 69% → **DPO 97%** | 「避免退化」是低階特徵 → DPO 真的**類推**到沒見過的題 |
| **topic**（難）| 對題定義 vs 張冠李戴（別條目的定義）| SFT 0% → DPO 9% | 需要「標題↔內容」語義綁定 → 8M char 模型學不動、只會**背** train |

> ⚠️ 兩種軸的 **train-acc 都會衝到 100%**——唯一差別在 held-out 能不能類推。這正是 MLOps
> 鐵則：**看 held-out、別看 train**。`make dpo` 把 held-out 偏好率記進訓練曲線、`make eval-dpo`
> 畫成 `artifacts/dpo_generalization.png`，一眼看穿「死背 vs 真學」。

**兩個「選對指標」陷阱（延續 SFT 評估的教訓）**：
- **長度偏誤**：用「總和 logπ」比 chosen/rejected 會被長度汙染（答案越長越吃虧），SFT 在 topic 軸
  量到 0% 正是這個假象 → 要用「**每 token 平均 logπ**」才量的是內容、不是長度。
- **fluency ≠ correctness**：base 模型覺得「通用但跑題」的定義比「具體但正確」的更順（per-token
  機率更高）——所以 topic 軸對小模型是 hard mode，要它違背語言機率先驗去偏好正確內容。

> SFT/DPO 模型都**不 promote 上 production**（它們是 chat/對齊版，評估尺不同——指令行為、偏好
> 類推，而非預訓練 perplexity）。production 仍是 8000 步那顆 base。

**精修：β 旋鈕（`make dpo-beta`）——一個被資料推翻的直覺**。β 是 DPO 唯一核心旋鈕，量三個取捨面：

| β | held-out 偏好 | 對 SFT 漂移（每 token \|Δlogπ\|）| 生成重複率 |
|---|---|---|---|
| 0.02 | 100% | **10.08** | 6% |
| 0.1 | 97% | 2.88 | 5% |
| 0.5 | 92% | **2.07** | 6% |

直覺以為「β 大＝KL 罰得重＝緊貼 reference」，但固定步數下**剛好相反**：DPO loss = −logσ(β·margin)
飽和在 **margin≈1/β** → β 越**小**，要追的 logπ gap 越大 → 漂移越大、越過度優化。所以
**小 β 不是更保守，是更激進**。甜蜜點在中間（β=0.1：偏好 97%、漂移溫和）。
**行為兌現**：DPO 把實際生成重複率從 SFT 的 8% 降到 5–6%——偏好不只贏在 logπ 排名，真的改變了產出。

### 10. RLHF：把 DPO 收合掉的東西拆開——reward model + RL，與招牌坑 reward hacking

DPO 是「免 reward model、免 RL」的捷徑。要真的懂 RLHF，得把它拆開成原本的兩階段，
親眼看到 DPO 省掉了什麼、以及為什麼那些零件會出包：

| 階段 | 做什麼 | 關鍵檔案 |
|---|---|---|
| **Reward Model**（①）| GPT 骨幹 + 純量 head，Bradley-Terry 損失學「好回答得高分」 | `src/reward_model.py`、`pipeline/07_reward_model.py`、`make reward` |
| **GRPO**（②）| 取樣一組 K 回答→RM 打分→組內標準化當 advantage→PG + KL 錨 | `pipeline/08_grpo.py`、`make grpo` |

> 用 **GRPO**（DeepSeek）而非 PPO：不需 critic/value 網路，是現代更簡單的 RL 對齊。
> `loss = −E[advantage·logπ(回答)] + β·KL(π‖π_ref)`，`advantage =（reward−組平均）/組標準差`。

**核心對照——reward hacking（`make grpo` 跑兩組、`make eval-grpo` 出圖）**。同時量兩把尺：
RM 分數（**代理**，RL 在最大化）vs 生成多樣性（**真實**品質，RM 沒直接管）：

| KL 錨 | RM 分數（代理）| 對 SFT 漂移 KL | 生成多樣性（真實）| 結果 |
|---|---|---|---|---|
| **ON** (β=0.05) | 3.7 → 3.5（穩）| ~1 | 100% → 100% | 受控、不退化 |
| **OFF** (β=0) | 3.7 → **13.2**（暴漲）| → **186** | 100% → **6%** | **reward hacking** |

無 KL 錨時，policy 把 RM 分數衝到 3.5×，但真實品質**崩**：mode collapse 成「不管問什麼都吐
同一串」——

```
問：什麼是哲學？   → 方言，的方言，的方言，的方言，的语句。
問：什麼是心理学？ → 方言，的方言，的方言，的方言，的语句。   ← 三題一模一樣
問：什麼是運算數學？→ 方言，的方言，的方言，的方言，的语句。
```

這串重複垃圾被 RM 打 +13 高分，因為它落在 RM **訓練分布外**（RM 只見過「中国中国」式重複）、
v_head 對它外推成高分 → policy 鑽到這個盲點。**RM 是學來的代理，優化過頭就被它的漏洞反咬**
＝ Goodhart's law：指標一旦變成目標，就不再是好指標。**KL 錨（β>0）把 policy 拉在 reference
附近、不准它跑去鑽漏洞，正是 RLHF 防 reward hacking 的關鍵**。這也是整個專案方法論的縮影：
**質疑你的指標有沒有被汙染/被鑽**。

### 11. IPO：把 DPO 的「margin 推到無窮」改成「回歸固定目標」（治過度優化）

DPO 用 $-\log\sigma(\beta\cdot m)$，σ 飽和後梯度雖小卻**永不歸零** → margin 被一路推爆（實測衝到
200+），是 §6 β-sweep 看到的過度優化的根。**IPO**（Azar 2023）改用平方損失
$(m-\tfrac{1}{2\beta})^2$，給 margin 一個**有限目標** $1/(2\beta)$，到了就停——天生防過度優化。
這正是 [`derivations.md`](docs/derivations.md) §3「margin 尺度≈1/β」的「把它釘死」版本。`make dpo-ipo` 出圖：

| 損失 | margin | held-out（format 軸）|
|---|---|---|
| DPO (β=0.1) | 130 → **204（爆衝）** | **97%** |
| IPO target=5 (β=0.1) | 釘在 ~5–12 | 78% |
| IPO target=10 (β=0.05) | ~18 | 80% |
| IPO target=25 (β=0.02) | ~25 | 84% |

兩個誠實的收穫：① **IPO 的 margin 受控**（釘在目標、不爆），target $1/(2\beta)$ 是顯式旋鈕——放大
（β 變小）偏好更強、held-out 爬升。② 但在**這個乾淨、值得用力學的偏好**（避免退化）上，DPO 的
激進其實**無害**，所以 DPO 97% > IPO 84%。**IPO 的勝場在「過度優化會傷的場景」**（偏好有雜訊/
spurious 時，DPO 會死記，IPO 的有限目標擋得住）——這裡偏好乾淨，所以是 IPO 拿穩定換掉一點準度。

### 12. PPO：補完 RL 家族——看 GRPO 到底簡化掉了什麼

GRPO 是 PPO 的精簡版。把 PPO（InstructGPT/ChatGPT 用的經典 RLHF）親手做一遍，就看得到
GRPO 丟掉了哪兩塊：

| 零件 | PPO 有 | GRPO 怎麼省掉 |
|---|---|---|
| **Critic / value 網路** $V_\phi(x)$ | 學一個基準，advantage = $R-V(x)$ | 改用「同組 K 個回答的平均獎勵」當基準（見 derivations §4：組內平均也是不偏 baseline）|
| **Clipped surrogate**（截斷目標）| ratio $\rho=\pi_\theta/\pi_{\theta_\text{old}}$ 夾在 $[1\pm\epsilon]$，可重用 rollout 訓多 epoch | GRPO 一批只訓一次，不需要 ratio/clip |

`pipeline/09_ppo.py`（`make ppo`）實作 critic + clipped 目標，並做 **clip vs 無 clip 對照**（關掉 KL 罰
$\beta=0$ 以隔離 clip 的作用、用較大 lr 與 8 epochs）：

| | RM 分數 | 生成多樣性 | 結果 |
|---|---|---|---|
| **clip** ε=0.2 | 6 → 11（穩定爬）| 100% → 100% | 更新被夾住、policy 穩 |
| **無 clip** | 7.5 →（崩）**0.2** | 100% → **6%** | 一步走太遠 → **policy 走壞、mode collapse** |

無 clip 時，「同一批 rollout 訓 8 個 epoch」讓某一步把 policy 推太遠 → 直接走壞（多樣性崩到 6%、
連 RM 分數都崩）。clip 把 importance ratio 夾住、限制單次更新幅度 → policy 穩定改進。這就是 PPO
名字裡 **Proximal（近端）** 的意義：**逼著新 policy 待在舊的附近，別一步跳太遠**。

> 至此 RL 對齊家族走完：**PPO（critic + clip，重）→ GRPO（丟 critic、用組內 baseline，輕）**；
> 偏好優化家族：**DPO（封閉式）→ IPO（有限目標）**。四種都親手做過、都有對照實驗。

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
│   ├── reward_model.py    # RLHF reward model（GPT 骨幹 + 純量 head + BT 損失）
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
│   ├── 05_sft.py          # 後訓練①：SFT 指令微調（base → 對話格式）
│   ├── 06_dpo.py          # 後訓練②：DPO / IPO 偏好對齊（--loss，policy/reference + held-out 監控）
│   ├── 07_reward_model.py # 後訓練③a：reward model（Bradley-Terry）
│   ├── 08_grpo.py         # 後訓練③b：GRPO RL（KL 錨 + reward hacking 偵測）
│   ├── 09_ppo.py          # 後訓練③c：PPO（critic + clipped 目標；clip vs 無 clip 對照）
│   ├── plot_loss.py       # loss 曲線（多 run 疊圖）
│   └── viz_attention.py   # attention 熱圖（單張 / --grid 全部 head）
├── scripts/
│   ├── get_data.sh        # 下載樣本語料
│   ├── make_messy_corpus.py # 產髒語料示範清洗/去重
│   ├── train_bpe.py       # 跑 BPE + 可審核監控（合併 log / report）
│   ├── rope_extrapolation.py # RoPE 外推 demo（train@64, eval→256）
│   ├── multi_seed.py      # B④ 多 seed 嚴謹：mean±std + 誤差線 + 真差異判定
│   ├── make_sft_data.py / eval_sft.py   # SFT 自抽問答資料 / 正規評估
│   ├── make_dpo_data.py / eval_dpo.py   # DPO 偏好對（topic/format 兩軸）/ 類推評估+出圖
│   ├── dpo_beta_sweep.py  # DPO 精修：掃 β 旋鈕（偏好/漂移/生成重複三取捨）
│   ├── eval_grpo.py       # RLHF 評估：reward hacking 對照圖（代理漲 vs 真實崩）
│   ├── dpo_vs_ipo.py / eval_ppo.py   # IPO 對照圖 / PPO clip-vs-無clip 對照圖
│   └── verify.py          # 驗收 playbook 執行器
├── serve/app.py           # 推論 API（FastAPI）：/health /generate /model /metrics
├── monitoring/            # Prometheus + Grafana stack（podman pod）
│   ├── prometheus.yml     #   抓 API /metrics
│   ├── grafana/           #   datasource + dashboard 自動 provisioning
│   └── up.sh / down.sh    #   一鍵起/收監控 stack
├── registry/              # 模型治理（審計軌跡，進 git）
│   ├── registry.json      #   模型台帳（digest / lineage / 狀態）
│   └── cards/<digest>.md  #   每顆模型的 model card
├── notebooks/
│   ├── 01_explore_data.ipynb # 資料探索
│   └── 02_monitor.ipynb      # 統一監控面板（BPE / 訓練 / 資料品質 / before-after）
├── tests/test_data.py     # 單元測試（資料 / 現代零件 / 取樣 / KV-cache / 治理 gate / API）
├── Containerfile          # 推論服務容器（GPU via CDI，模型 mount）
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

# 資料品質報表（偵測器掃全語料 → 命中%/門檻/問題樣本）
make quality       # 輸出 data_quality_report.json + 印表格

# 部署下半場（MLOps serving）
make serve         # 起推論 API（http://127.0.0.1:8000/docs）
make image         # podman build 推論服務 image（GPU）
make run-container # GPU 跑容器（模型 mount 進去）
make dashboard     # 起 Prometheus+Grafana 監控 stack（http://127.0.0.1:3000）
make register      # 把目前模型註冊進 registry（產 model card）
make models        # 看 model registry 台帳
python scripts/registry_cli.py promote <digest前綴>   # 升 production（要過 gate）

# 進階（e）
make retrain       # 重訓迴圈（資料→訓練→評估→註冊→gate）；--auto-promote 自動上線
make compress      # fp16/int8 量化對比（大小 vs 品質）
CANDIDATE_CKPT=/path/ckpt.pt CANARY_PCT=20 make serve   # 金絲雀：20% 流量導候選
BATCH_MAX=8 make serve                                  # 動態批次：衝吞吐

# 後訓練（對齊：把接龍機器變助理）
make sft-data && make sft && make eval-sft   # 里程碑1：SFT 指令微調 + 正規評估
make dpo-data && make dpo && make eval-dpo   # 里程碑2：DPO 偏好對齊 + 類推 vs 死背曲線圖
make dpo-beta                                # DPO 精修：掃 β 旋鈕（偏好/漂移/生成重複三取捨）
make reward && make grpo && make eval-grpo   # 里程碑3：RLHF（reward model + GRPO）+ reward hacking 對照圖
make dpo-ipo                                  # 精修：DPO vs IPO（margin 爆衝 vs 釘在 1/2β、治過度優化）
make ppo && make eval-ppo                      # 補完 RL 家族：PPO（critic+clip）；clip vs 無 clip→走崩 對照圖
```

## 學習弧線（本專案的設計脈絡）

B 監控 → D 原理 → A 調參 → C tokenizer，口訣「B 給眼睛、D 給腦、A 動手、C 升級」：

- **B 監控**：loss 曲線、過擬合 gap、統一監控面板（先有儀表才好做實驗）
- **D 原理**：self-attention（Q/K/V/熱圖）、causal mask、multi-head、residual
- **A 調參**：context window、模型大小，邊改邊看曲線（見上方關鍵數據）
- **C 升級**：char → BPE subword tokenizer，並學會用 BPC 公平比較

## See Also

- `docs/lessons-learned.md` — ⭐ 關鍵時刻與方法論教訓（「以為對、量了才發現錯」的瞬間，含 SFT 評估兩個陷阱）
- `docs/case-study.md` — 給人讀的完整旅程敘事（作品集入口）
- `docs/interview-talking-points.md` — 面試 20 分鐘怎麼講（口袋小抄 + 4 個 STAR 故事 + 預期問答）
- `docs/theory-map.md` — 程式碼 ↔ 論文/聖經本 三層對照地圖（地基→現代架構/效率/取樣→後訓練對齊，每項附出處與年份、標新舊）
- `docs/reading-the-charts.md` — 讀圖指南：每張關鍵圖表「在說什麼」怎麼看（通用三步驟＋後訓練四張圖）
- `docs/derivations.md` — ⭐ 關鍵演算法的**數學推導/證明**（RLHF→DPO 封閉式、margin≈1/β、policy gradient、softmax-CE、RoPE 相對位置、√d、MinHash=Jaccard），每條附 Java 直覺與程式碼對應
- `docs/learning-plan.md` — 即時深挖學習清單（哪些深挖、哪些知道即可）
- `docs/data-pipeline.md` — 資料這塊各階段在做什麼
- `docs/verification-playbook.md` — 驗收 runbook
- Karpathy, *Let's build GPT: from scratch* / **nanoGPT**（本專案精神來源）
- Vaswani et al. 2017, *Attention Is All You Need*；Sennrich et al. 2016（BPE）
