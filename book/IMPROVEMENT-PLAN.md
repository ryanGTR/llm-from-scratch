---
title: 書籍改進計畫（斷 session 可接續）
type: plan
created: 2026-06-22
updated: 2026-06-22
tags: [book, plan, handoff, quarto]
---

# 書籍改進計畫

> **接續者先讀這頁。** 這是把目前的書（`book/`，已發佈 <https://ryangtr.github.io/llm-from-scratch/>）
> 從「B+ 的延伸技術敘事」提升到「A 級成書」的計畫。包含現狀評論、每個缺點的**具體補法**、
> 一份「A 級章節檢查表」、各章現況、和下一步。

## ⏩ 目前狀態（最後更新 2026-06-22，接續者先看這格）

- **🆕 價值槓桿（第六段）：對外 baseline 對照（Ch5）。** 補掉「評估都跟自己比」的可信度缺口：
  新增 `book/examples/tiny_baseline.py`——tiny GPT vs 亂猜 vs gzip vs 4-gram(backoff)，同一份 val、
  同一把尺 BPC（純 CPU 真跑）。實測 **亂猜 6.02 → gzip 3.19 → 4-gram 2.55 → tiny GPT 2.32**：
  神經網路贏過會數頻率的 n-gram＝賺到複雜度（但只贏 0.23＝笨基準很硬、要記得跑）。Ch5 加一節
  「先贏過笨基準」(@sec-baseline) + `baseline_bpc.png` + 「選下界基準不選虛榮基準(GPT-2)＝選對 baseline」
  的 callout。術語表加「笨基準」、`make book-smoke` 納入、`tests/test_book_examples.py` 加 2 條不變量
  （make test 58 綠）。**誠實界線寫進書：這是對的尺度的 baseline（證明賺到複雜度+尺站得住），不宣稱
  打贏真 LLM**。下一個更大的價值槓桿仍是「真實規模故事」（非再加輔助）。
  - **②正確性 baseline 也補了**：`tiny_correctness.py` 把手刻 causal attention 對 torch 內建
    `scaled_dot_product_attention` 數值對拍（單頭 2.38e-07／多頭 4.77e-07＋結構不變量），Ch5
    加 `### 正確性 baseline：你沒在騙自己`(@sec-correctness)，接第 3 章 FlashAttention 同一招。
    進 make book-smoke、tests 加 1 條（make test 59 綠）。
- **🆕 可讀性（第五段）：給「技術主管」的定位輔助。** 因為這是「LLM × MLOps × 治理」一半一半的
  領域、讀者只精通一邊，加了三層 orientation 讓不熟那邊的主管也不迷路：
  - **前言全書地圖**（Mermaid `@fig-bookmap`）：橫向生命週期 + 促上線/重訓回饋迴圈 + 後訓練分支。
  - **每章「你在這裡」地圖**（全 9 章）：同一條 9 節點生命週期，當前章紅底點亮，放章首一句話後。
  - **每章「🎯 給技術主管：關鍵術語速查」框**（全 9 章，可摺疊）：每術語一句白話 + 「為什麼你該在意
    （成本/風險/治理）」，不教實作、熟的人可跳過。
  - 用 **Mermaid**（diagrams-as-code、CJK 標籤乾淨、進 git，比 PNG 好維護）；render 全綠。
- **🆕 重大重構（第四段）：MLOps 變成書的主軸。** Ryan 點出「專案主軸是 MLOps，但書裡只佔 Ch6
  一章、變配角」。已重排成四部、把原 Ch6 一章拆成三章、MLOps 升為「第三部（重心）」：
  - 第一部 原理：01 最小GPT・02 現代零件・03 效率
  - 第二部 資料與評估：04 真實資料・05 評估
  - **第三部 維運 / MLOps（重心）：06 服務化與可觀測・07 模型治理・08 會腐壞的系統**
  - 第四部 對齊：09 對齊（原 07）
  - 附錄：10 數學（原 08）・11 術語表（原 09）・99 references
  - 新增 2 支 example：`tiny_observability.py`（冷啟動/p50/p95）、`tiny_drift.py`（PSI 抓漂移）；
    新增 2 張真跑插圖：`serving_latency.png`、`drift_psi.png`。前言加「這本書的重心：MLOps」段。
  - 交叉參照用 label 不靠檔名，改檔名/拆章後 render 仍零 unresolved；新章標籤：`#sec-serving`
    `#sec-build-service` `#sec-tiny-observability` / `#sec-governance`(沿用) `#sec-build-governance`
    `#sec-tiny-serve` / `#sec-drift` `#sec-build-drift` `#sec-tiny-drift`。
- **已深耕到 A 級**：✅ 全 9 章（01–09）+ 附錄 + 術語表。每章都有：學習目標框＋一段逐行走讀＋
  一個 💻 純 CPU 真跑範例（真數字）＋章末 3 題（預測/動手/弄壞，答案摺疊）。
- **下一步**：➡️ 全書主體 + 收口 + 插圖 + CI + MLOps 重構都完成。剩 Ryan 拍板 push + 重部署 GitHub Pages。
- **✅ 本波（2026-06-22 第三段）插圖 + CI**：
  - **插圖**（4 張，全部由真跑數據產生、可重現）：`make_book_figures.py` 產出 →
    Ch3 `kvcache_speedup.png`、Ch4 `dedup_blindspot.png`、Ch6 `governance_gate.png`、
    Ch7 `dpo_trainvsheldout.png`，各章已用 `@fig-` 嵌入 + 中文圖說。`make book-figures` 一鍵重產。
  - **examples 接進 CI**：每支加 `BOOK_SMOKE` 環境開關（縮成極小步數）；新增 `make book-smoke`
    端到端跑通 7 支；`tests/test_book_examples.py`（kvcache 對拍相同 + dedup 三不變量，~1s）併進
    `make test`；`ci.yml` 加一步 `make book-smoke`。
  - **順手修**：`tiny_kvcache.py` 的 cache 無滑動視窗，`n_new>context` 會爆 → context 512→1024
    （章末習題 n_new=1000 才不掛）；Ch3 內嵌數字同步更新（10.13×、0.95M、context 1024）。
    `tiny_dedup.py` 重構成可 import（函式化 + `main()` guard），CI/出圖共用。
- **✅ 本波收口（2026-06-22 第二段）**：
  - **09 術語表**（#6）：新建 `09-glossary.qmd`（符號約定表 + ~50 條術語，每條一句話 + 出處交叉連結），
    已加進 `_quarto.yml` 附錄、render 通過（52 個 xref 全解析）。
  - **前言受眾段**（#4）：`index.qmd` 加「這本書寫給誰、你需要先會什麼」（有經驗工程師 + ML 新手；
    需 Python + 線代 + 一點機率；不需 ML/GPU）。
  - **Ch8 補直覺**（#5）：四條推導各加一個「直覺」callout（兩步壓一條 / β 是反方向煞車 /
    MinHash 是便宜估計 / baseline 是跟平均比），並把政策梯度節補上 `{#sec-pg-math}` 標籤、
    全部交叉連到對應章末習題與 💻 走讀。
- **本機 commit 但⚠️尚未 push**：`5f4962b`(Ch1)、`394f2d6`(Ch2)、`<本波 ch3-7>`。
  Ryan 還沒決定要不要 push + 重部署 GitHub Pages。
- **本波順手修的全書級 bug**：`08-math-appendix.qmd` 標題原為 `.unnumbered`，導致全書對
  `@sec-dpo-math / @sec-margin-math / @sec-minhash-math / @sec-rope-math` 的交叉參照**全部解析失敗**
  （render 時 `Unable to resolve crossref`、HTML 連結變成同頁死錨、文字顯示成 `sec-margin-math`）。
  拿掉 `.unnumbered` 後 8 處 -math 參照全部解析成 `08-math-appendix.html#sec-...`（顯示「小节 8.x」），
  render 零警告。附錄因此變成有編號（章 8），可接受。
- **配套程式新慣例**：A 級章節的「💻 在你的機器上」對應 `book/examples/*.py`，全部純 CPU、真跑過——
  - `tiny_gpt.py`（Ch1）：自包含 char-level 小 GPT，~1分50秒，val 1.59。
  - `tiny_modern.py`（Ch2）：一鍵切 RMSNorm/SwiGLU/RoPE 對照，~7分鐘。
  - `tiny_kvcache.py`（Ch3）：KV-cache vs naive，先對拍逐 token 相同（證對）再計時，~十幾秒、不需語料（9.58×）。
  - `tiny_dedup.py`（Ch4）：注入重複+髒語法，示範聚合指標盲點(20.7%)＋MinHash/LSH(候選佔1.9%)，純標準庫、不需 torch、幾秒。
  - `tiny_eval.py`（Ch5）：亂猜基準 ln65=4.174、BPC、3-seed mean±std(1.680±0.005)，~3–4分鐘。
  - `tiny_serve.py`（Ch6）：sha256 digest 身份(認內容不認檔名)＋promotion gate(BLOCK/PROMOTE/UNREGISTERED)，~1分鐘。
  - `tiny_dpo.py`（Ch7）：DPO loss 逐行＋兩種偏好軸示範「train-acc 騙你」（A 100%/100%、B 100%/47.7%），~2–3分鐘。
  - 需語料的都讀 `input.txt`（= repo 的 `data/raw/input.txt`，1.1MB tiny shakespeare；examples 內已 `ln -s`）。
  - **鐵則：書裡嵌的 loss/數字/生成樣本都要「真的在 CPU 上跑過」再貼，不杜撰**（合 Ryan「如實回報 / 選對指標」）。
  - **可重現性雷**：MinHash 別用 Python 內建 `hash()`（對字串會加鹽、跨 process 不一致）——用 `zlib.crc32`。
- **每章 A 級配方**（套到下一章）：①開頭學習目標框（假設你已會什麼 / 學完會做什麼）②一段「逐行建起來」走讀
  ③一個 💻 CPU 最小可跑版（真跑、真數字）④章末 2–3 題（先預測 / 動手 / 弄壞，答案用 `::: {.callout-tip collapse="true"}` 摺疊）。
- **render 驗證**：`cd book && ~/.local/quarto/bin/quarto render 0X-xxx.qmd --to html`（quarto 在非系統 PATH）。
- **本機預覽**：`~/.local/quarto/bin/quarto preview --no-browser --port 4321`（改檔自動刷新）。

## 目前定位（誠實評論）

**B+ 的「作品集深水區 / 延伸技術敘事」，不是 A 的「成書」。** 當作品集/履歷非常夠用；當要出版
還有一段。

- **強項**：① 有別人沒有的方法論主線（predict→量→打臉→懂 + 質疑你的尺）② 「假設被打破」的瞬間
  （reward hacking、KV-cache 反而慢、21.6% 偵測器、DPO 死背）是全書最強 ③ 治理/稽核視角獨家
  （Goodhart / promotion gate / digest 接銀行本行）——最該放大的賣點。
- **弱項**：① 深度不均（Ch1/7/8 紮實、Ch2–6 偏摘要）② 太依賴 repo（一堆「見 repo / make X」）
  ③ 程式碼是片段非走讀 ④ 受眾沒明講 ⑤ 練習太少。
- **缺口**：完整 worked example、術語表/符號/索引、wow moment、讀者重現性。

## 每個缺點 → 具體補法

| # | 缺點 | 怎麼補（具體動作） |
|---|---|---|
| 1 | 深度不均（Ch2–6 偏摘要）| 每章補一個「**從零做一次**」小節：完整程式碼 + 在你機器跑的指令 + 預期數字 + 怎麼解讀 |
| 2 | 太依賴 repo | 把「見 repo / make X」改成**完整內嵌程式碼**；加 callout「💻 在你的機器上」給**不用 GPU 的最小版**（1 MB 莎士比亞而非 105 MB 中文，讀者筆電可重現）|
| 3 | 程式碼片段非走讀 | 改成「**逐行建起來**」：空殼 → 一步步填 → 每步解釋。例：Ch1 attention 從 `q@k` 建到完整 forward |
| 4 | 受眾沒明講 | 前言加「**這本書寫給誰 + 你需要先會什麼**」：給「有經驗的工程師、ML 新手；需 Python + 線代基礎，不需 ML 背景」 |
| 5 | 練習太少 | 每章末加 2–3 題（**先預測題 / 動手題 / 弄壞題**），答案摺疊；複用 `docs/exercise-dpo-derivation.md` 格式 |
| 6 | 缺術語表/符號/索引 | 新增附錄 `09-glossary.qmd`（術語表 + 符號約定）；Quarto 可自動產索引 |
| 7 | wow moment 不夠 | Ch4 或 Ch7 嵌並排方塊「base 亂寫 vs SFT 會應答 vs 真實生成中文」，讓讀者親眼看到模型在做事 |
| 8 | 讀者重現性 | 同 #2：每個重實驗給一個筆電版（CPU + 1 MB 資料）的縮小複本 |

## 「A 級章節」檢查表

一章算「深耕到 A 級」要滿足：

- [ ] 開頭講清楚「這章假設你已經會什麼、學完你會做什麼」
- [ ] 至少一段**逐行走讀**的程式碼（不是貼片段）
- [ ] 一個 **💻 在你的機器上** 的最小可跑版（CPU / 1 MB 資料 / 幾十秒）
- [ ] 關鍵程式碼**完整內嵌**，讀者不 clone 也讀得懂
- [ ] 數學（若有）附直覺，不只丟式子
- [ ] 章末 2–3 題（先預測 / 動手 / 弄壞），答案摺疊
- [ ] 至少一個「假設被打破」或 wow 的具體瞬間

## 各章現況

| 章 | 現況 | 待辦 |
|---|---|---|
| 前言 | **✅** | 已加「受眾＋假設知識」段（#4）|
| Ch1 最小 GPT | **✅ A 級（範本）** | 已補：學習目標框、attention 逐行走讀（@sec-build-attention）、💻 CPU 版自包含 `tiny_gpt.py`（真跑、真數字 val 1.59、莎士比亞 wow 對照）、章末 3 題（預測/動手/弄壞，答案摺疊）。**其他章照這個範本套** |
| Ch2 現代零件 | **✅ A 級** | 已補：學習目標框、💻 CPU 對照（`tiny_modern.py` 一鍵切 RMSNorm/SwiGLU/RoPE，真跑 5 配置驗「準/省」：RMSNorm +0.003 持平、SwiGLU −0.019、RoPE −0.007、全開 −0.025 且省參數）、章末 3 題（預測 GQA/對答案/弄壞）。保留準vs省主線+repo 表+RoPE 外推圖 |
| 03 效率取樣 | **✅ A 級** | 學習框、KV-cache 逐行(@sec-build-kvcache)、💻 `tiny_kvcache.py`(對拍證對再計時 10.13×)、章末 3 題、`kvcache_speedup.png` |
| 04 真實資料 | **✅ A 級** | 學習框、MinHash/LSH 逐行(@sec-build-dedup)、💻 `tiny_dedup.py`(聚合盲點20.7%＋LSH候選1.9%)、章末 3 題、`dedup_blindspot.png` |
| 05 評估 | **✅ A 級** | 學習框、BPC 逐行(@sec-build-bpc)、💻 `tiny_eval.py`(亂猜基準＋BPC＋3-seed std=0.005)、章末 3 題 |
| **06 服務化與可觀測** | **✅ A 級（新，MLOps）** | 學習框、帶 metrics 服務逐行(@sec-build-service)、💻 `tiny_observability.py`(冷啟動+p50/p95)、章末 3 題、`serving_latency.png` |
| **07 模型治理** | **✅ A 級（moat，MLOps）** | 稽核四問、digest+gate 逐行(@sec-build-governance)、💻 `tiny_serve.py`(digest不認檔名＋gate 真擋)、章末 3 題、`governance_gate.png` |
| **08 會腐壞的系統** | **✅ A 級（新，MLOps）** | drift/重訓/canary/shadow/放量、PSI 逐行(@sec-build-drift)、💻 `tiny_drift.py`(PSI 抓漂移+放量坑)、章末 3 題、`drift_psi.png` |
| 09 對齊（原 07）| **✅ A 級（皇冠）** | 學習框、DPO loss 逐行從 seq_logprob(@sec-build-dpo)、💻 `tiny_dpo.py`(train-acc騙你)、章末 3 題、`dpo_trainvsheldout.png` |
| 10 數學附錄（原 08）| **✅** | 四條推導各加直覺 callout + 政策梯度補 `{#sec-pg-math}` + 交叉連到章末習題；`.unnumbered` 已移除修交叉參照 |
| 11 術語表（原 09）| **✅ 已建** | 符號約定 + ~50 條術語（含 PSI/observability/data drift→新 MLOps 章）（#6）|

## 下一步（建議）

**✅ Ch1 已深耕成 A 級範本**（見上表）。配套程式 `book/examples/tiny_gpt.py`——自包含、純 CPU、
真跑過（0.62M 參數、3000 步、val loss 1.59、生成出有莎士比亞形狀的對白）。**接續者照這個範本套下一章。**

**已完成**：Ch1–Ch7 全部深耕成 A 級、前言受眾段(#4)、Ch8 補直覺(#5)、09 術語表(#6)——A 級檢查表全項收口。
**驗證方式**：每章 `~/.local/quarto/bin/quarto render --to html`（或全書 render）→ 確認零 error、
零 `Unable to resolve crossref`、新 anchor 都在、嵌入數字與 examples 真跑輸出一致。本波已驗過。
範本要點：每章都該有①學習目標框 ②一段逐行走讀 ③一個 💻 CPU 最小可跑版（真跑、真數字）
④章末 2–3 題（預測/動手/弄壞，`::: {.callout-tip collapse="true"}` 摺疊答案）。

## 怎麼重新部署（改完之後）

見 `book/README.md` 的「重新部署到 GitHub Pages」一節（render HTML → worktree 推 gh-pages）。
Quarto 在 `~/.local/quarto/bin/quarto`（非系統 PATH）。
