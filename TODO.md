# TODO — 用現代技巧把「玩具」修補成更接近真實

這份是從專案的已知缺陷整理出的修補路線圖。每項標了「對應的最新技巧」與難度。
原則延續專案：**先預測、動手做、看監控數據、用測試/playbook 守住**。

勾選圖例：難度 🟢 小 / 🟡 中 / 🔴 大

---

## 1. 把模型零件升級成「現代 LLM 同款」（最新架構技巧）

目前 `model.py` 是 2017 原版 Transformer 零件。換成 2023+ 主流（LLaMA/Mistral 同款）：

- [ ] 🟡 LayerNorm → **RMSNorm**（更省、現代標配）
- [ ] 🔴 學習式 position embedding → **RoPE（旋轉位置編碼）**（外推更好，現代主流）
- [ ] 🟡 GELU MLP → **SwiGLU**（gated，效果更好）
- [ ] 🔴 Multi-head → **GQA（grouped-query attention）**（省 KV-cache，Llama-2/3 用）
- 對應缺陷：技術債「零件是舊版」。做完 `model.py` 就是「現代 LLM 同款骨架」。

## 2. 效率：推論/訓練加速（最能立刻見效）

- [x] 🟢 **FlashAttention**（torch `F.scaled_dot_product_attention`）✅ `use_flash`；數學等價、記憶體 O(T²)→O(T)、context 上限實測 ×4
- [x] 🔴 **KV-cache**（生成時每步只算新 token，O(T²)→O(T)）✅ `generate(use_kv_cache=True)`；驗證與不快取完全相同、~2.2× 快
- [x] 🟢 取樣 **top-p / min-p**（自適應截斷）✅
- 對應缺陷（軸一「要不要快取」）已解。

### 2b. KV-cache 壓縮（軸二：把快取「變小」，省 VRAM；三條正交手法）

- [x] **砍頭數** MQA/GQA（少存幾組 k/v）✅ 已做（`n_kv_head`；MQA=設 1）。Shazeer 2019 / Ainslie 2023（Google）
- [ ] 🔴 **低秩潛在 MLA**（把每個 KV 壓成更小的潛在向量）— DeepSeek-V2 2024
- [ ] 🔴 **量化 TurboQuant**（k/v 形狀不變，每數字 16-bit→~3.5-bit，4.5–5× 壓縮、準度幾乎不掉）— Google 2026。做法：隨機 Hadamard 旋轉消離群值 + 1-bit QJL 校正殘差；隨機矩陣免 calibration。**正交於 GQA → 可疊在 GQA 上面再壓**。
- 註：三者正交，可組合（GQA × 量化 × 低秩）。我們已有 GQA，量化/低秩是進階。

## 3. tokenizer 補強

- [ ] 🟡 BPE 從 char-level → **byte-level**（處理任何 unicode/emoji，修「遇生字直接丟掉」的真 bug）
- [ ] 🟡 BPE 效率：純 Python O(N×merges) 太慢 → 增量計數，或直接接 **tiktoken**
- [x] 🟢 `.bin` 自適應 **uint16/uint32** ✅（vocab>65535 自動用 uint32，dtype 記進 meta，所有讀取端依此讀；解溢位）
- 對應缺陷：技術債「BPE 慢/char 起步」「uint16 上限」。

## 4. 實驗嚴謹度（讓數據站得住腳）

- [x] 🟡 多 seed 重跑（mean ± std + 誤差線）✅ `scripts/multi_seed.py`：確認 SwiGLU/RoPE 真差異、雜訊地板≈0.01
- [x] 🟢 獨立 **test set** ✅（train/val/test 三切；`--test_frac`、`03_eval --split test`）
- [x] 🟡 **Deep Ensemble** ✅ `scripts/deep_ensemble.py`：3 個不同 seed 模型機率平均 → val 1.799 < 最佳單一 1.835（降 0.035，「免費」提升，隨機森林的神經網路版）。
- 對應缺陷：「實驗不嚴謹」。多 seed 已做；test set / deep ensemble 待補。

## 5. MLOps（工程化往生產級靠）★ Ryan 的戰略甜點

MLOps = DevOps 的 ML 版：把「notebook 訓一個模型」變成「可重現、可靠、能持續
運作的生產系統」。比 DevOps 多了「資料」與「模型會腐壞（data drift）」兩個變數，
所以模型永遠不算做完，要持續監控+重訓。**這條線正好接 Ryan 的企業 IT / 治理 /
CI-CD / IDP-CaaS 既有技能——走「ML 平台 / MLOps 工程師」用 ops 底子當差異化，
不必跟研究員拼數學/算力。** 專案已有「嬰兒版」，標 ✅。

整個生命週期：

- [ ] 🟡 **config 管理**：argparse → Hydra / OmegaConf（或 pydantic-settings），超參數集中可組合
- [ ] 🟢 **依賴鎖定**：`uv lock`（uv.lock）保證換機器可重現
- [ ] 🟡 **實驗追蹤**：W&B / MLflow，自動記每次 run 的參數+曲線（✅ 嬰兒版＝`runs/*.csv` + `02_monitor` 面板）
- [ ] 🟡 **資料 / 模型版本**：DVC，哪版資料訓出哪版模型、可追溯
- [ ] 🟡 **pipeline 編排**：Makefile → Airflow / Kubeflow / Prefect（✅ 嬰兒版＝Makefile）
- [ ] 🟡 **CI/CD for ML**：GitHub Actions，push 自動跑 `make test` + `make verify`（eval gate，品質沒過不上線）（✅ 嬰兒版＝verify playbook + test gate）
- [ ] 🟡 **model registry**：模型版本/升版/回滾、標記哪個在 prod
- [ ] 🔴 **部署 / 服務**：打包模型上線推論 API、KV-cache、批次、自動擴縮（vLLM）
- [ ] 🔴 **生產監控**：盯效能 / 資料漂移 / 延遲 / 成本，退化告警（ML 獨有）
- [ ] 🔴 **重訓迴圈**：偵測漂移 → 自動重訓 + 重部署
- 對應缺陷：「無 config/CI/lockfile/實驗追蹤」+ 生產級全套缺。
- 建議起手（最熟、最快有感）：**GitHub Actions CI** → uv lock → Hydra config。

## 6. 能力：放大（需要算力，不只是技巧）

- [ ] 🔴 放大模型/資料：上**雲端 GPU**（runpod / Lambda / vast.ai），pipeline 不用改、只換 device
- [x] 🔴 post-training：**SFT → DPO** ✅
  - SFT（里程碑1）：`pipeline/05_sft.py`，base→對話格式；正規評估 `eval_sft.py` 暴露兩個評估陷阱
  - DPO（里程碑2）：`pipeline/06_dpo.py`（policy+凍結 reference，封閉式偏好損失，免 reward model）。
    核心發現＝**容量內就類推、超出就只會背**：format 軸（連貫 vs 退化）held-out 69%→97% 真類推；
    topic 軸（對題 vs 張冠李戴）需語義綁定、8M 學不動只背 train（held-out ~9%）。`make eval-dpo` 出圖。
  - DPO 精修＝β 旋鈕掃描（`make dpo-beta`）：固定步數下「β 越小漂移越大」翻直覺；行為兌現＝生成重複率 8%→5–6%
- [x] 🔴 **RLHF（reward model + GRPO）** ✅
  - RM（`pipeline/07_reward_model.py`、`make reward`）：GPT 骨幹 + 純量 head + Bradley-Terry，held-out 偏好 100%
  - GRPO（`pipeline/08_grpo.py`、`make grpo`）：取樣→RM 打分→組內 advantage→PG + KL 錨，不需 critic
  - **核心對照＝reward hacking**：無 KL 錨(β=0) → RM 分數 3.7→13.2 暴漲，但生成多樣性 100%→6% mode collapse
    （三題全吐「方言，的方言…」高分垃圾，鑽 RM 分布外盲點）＝Goodhart；KL 錨(β>0) 防止之。`make eval-grpo` 出圖。
  - 下一步可選：DPO mask-prompt 精修 / IPO（防過度優化）/ 接雲端放大規模
- 對應缺陷：「本質是玩具/沒 post-training」——對齊**機制**（SFT→DPO→RLHF）已示範完整；**能力**仍受 8M 規模限。

## 7. 學習：換我主導（內化，不是看）

- [ ] 🟡 上面挑一塊，**Ryan 親手寫/debug，Claude 只在旁糾錯**（補「我懂≈我看懂」這個缺陷）

---

## 建議起手順序

1. **FlashAttention（2-①）**🟢 — 一行換掉、立刻看到 context 上限 ×4，回報最高
2. **多 seed 重跑（4-①）**🟡 — 讓既有數據站得住，也順便練實驗紀律
3. **現代零件 RMSNorm/SwiGLU（1）**🟡 — 把骨架升級成現代款，最有「跟上最新」的感覺
4. 之後再往 KV-cache / RoPE / 工程化 / post-training 推進

> 每做一項：先預測效果 → 改 → 看 `02_monitor.ipynb` 的曲線 → 跑 `make test` 守住。
