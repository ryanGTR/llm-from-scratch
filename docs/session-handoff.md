---
title: Session 接續點（斷 session 後從這裡繼續）
type: handoff
updated: 2026-06-22
---

> **最新（2026-06-22 整天，Telegram 驅動的大改一輪 — 書）**：把 `book/` 從「B+ 延伸敘事」做成 **A 級成書**，
> 並**重定位成 MLOps 為主軸**。依序（全部已 commit + **已 push origin/main + 線上書 gh-pages 重新部署 + CI 綠**）：
> ① Ch1–9 全章深耕 A 級（學習框 + 逐行走讀 + 💻 純 CPU 真跑範例 + 章末 3 題；數字皆真跑不杜撰）
> ② 收口：前言「受眾＋假設知識」段、Ch8(數學)四條推導補直覺、新增 **11 術語表**
> ③ **重大重構：四部、MLOps 升為「第三部 重心」**——原 Ch6 一章拆成 **06 服務化與可觀測 / 07 模型治理 / 08 會腐壞的系統**；
> 對齊移第四部(09)、附錄(10 數學/11 術語表)。檔名改了但交叉參照用 label 不靠檔名 → render 零 unresolved。
> ④ **真跑插圖 7 張**（`book/examples/make_book_figures.py` 從真數據生）+ **examples 接 CI**（`make book-smoke` 9 支端到端、
> `tests/test_book_examples.py`、ci.yml；**make test 59 綠**）
> ⑤ 「給技術主管」定位輔助：前言全書 Mermaid 地圖 + **每章「你在這裡」地圖** + 每章**「🎯 給技術主管」術語速查框**（可摺疊）；章首瘦身（學習框改可摺疊）；README 加「拿這個 repo 做什麼（依投入程度）」入口
> ⑥ **Ch5 兩個 baseline**：品質(`book/examples/tiny_baseline.py`：tiny GPT BPC 2.32 < 4-gram 2.55 < gzip 3.19 < 亂猜 6.02，@sec-baseline)
> ＋正確性(`tiny_correctness.py`：手刻 attention 對 torch SDPA 差 2e-7，@sec-correctness)。
> **書 1 接續＝先讀 `book/IMPROVEMENT-PLAN.md` 最上方「⏩ 目前狀態」（單一真相）。** 新增 examples：tiny_gpt/modern/
> kvcache/eval/dedup/observability/serve/drift/dpo/baseline/correctness（全純 CPU、CI 顧著）。Mermaid 在此 Quarto 原生可 render(CJK OK)。
>
> 📕 **決定寫第二本書＝邊緣運算（on-device + 邊緣機隊治理，第一本續集、雙倍下注治理 moat）**。**尚未起骨架**
> （Ryan 在想範圍）。記錄＝`book2-edge/IMPROVEMENT-PLAN.md`；第一部素材＝`book2-edge/research/slm-market.md`
> （SLM 應用+國際市場 deep-research 報告，有引用、對抗式驗證過）。前置：同 repo book2-edge/；無 ARM 硬體→edge 部署
> 在 CPU+llama.cpp/ONNX 並誠實標注「裝置模擬」；範圍待 Ryan 拍板（我建議先做第四部「邊緣機隊治理」）。
>
> 🔭 **我給 Ryan 的誠實評估**（他多次問）：以「金融×MLOps×治理 作品集/面試主菜」這把尺＝A−/A；核心天花板未動
> （8M 玩具＝展示判斷力非規模執行）；scaffolding 邊際報酬已到反曲點；**下一個真價值槓桿＝真實規模故事**，不是再加輔助。
> 定位＝「面試官友善的能力證明」，讀者＝金融/受監管 ML 技術主管。

> **前一波（2026-06-21 下午）**：完成**後訓練里程碑2＝DPO 偏好對齊**（`pipeline/06_dpo.py`、
> `make dpo`/`eval-dpo`、`tests/test_dpo.py` 5 條，全套 `make test` 39 條綠）。核心發現：
> 兩種偏好軸對照 → format（連貫 vs 退化）held-out **69%→97% 真類推**；topic（對題 vs 張冠李戴）
> 8M **學不動只背 train**（held-out ~9%）。圖 `artifacts/dpo_generalization.png`。
> **DPO 精修＝β 旋鈕掃描**（`make dpo-beta`、`scripts/dpo_beta_sweep.py`）：固定步數下「β 越小漂移越大
> （margin 目標≈1/β）」翻掉教科書直覺；行為兌現＝生成重複率 SFT 8%→DPO 5–6%。圖 `dpo_beta_sweep.png`。
> **後訓練里程碑3＝RLHF（reward model + GRPO）**：`src/reward_model.py`、`pipeline/07_reward_model.py`(`make
> reward`，BT 損失、held-out 偏好 100%)、`pipeline/08_grpo.py`(`make grpo`，GRPO=取樣→RM 打分→組內 advantage→
> PG+KL 錨，不需 critic)、`scripts/eval_grpo.py`(`make eval-grpo`)、`tests/test_rlhf.py` 5 條。**核心對照＝
> reward hacking**：無 KL 錨(β=0) RM 分數 3.7→13.2 暴漲但生成多樣性 100%→6% mode collapse（三題全吐「方言，
> 的方言…」鑽 RM 分布外盲點）＝Goodhart；KL 錨(β>0) 防之。圖 `grpo_reward_hacking.png`。
> **IPO 精修**（`pipeline/06_dpo.py --loss ipo`、`make dpo-ipo`、`scripts/dpo_vs_ipo.py`）：平方損失把 margin
> 回歸到目標 1/(2β) 不爆（vs DPO 衝 204）；clean 偏好下 DPO held-out 97% > IPO 78%（IPO 拿穩定換準度）。圖
> `dpo_vs_ipo.png`。
> **PPO 補完 RL 家族**（`pipeline/09_ppo.py`、`make ppo`、`scripts/eval_ppo.py`）：critic value 網路 + clipped
> surrogate（GRPO 簡化掉的兩塊）。clip vs 無 clip 對照（β=0 隔離）：無 clip 一步走太遠 → policy 走崩（RM
> 7.5→0.2、多樣性 100%→6%），clip 穩定爬＝Proximal。圖 `ppo_clip_vs_noclip.png`。`make test` 49 綠。
> 📘 **書（Quarto，已發佈 <https://ryangtr.github.io/llm-from-scratch/>）**：8 章+附錄都寫滿可讀。
> 要繼續**改進/深耕**這本書 → 先讀 `book/IMPROVEMENT-PLAN.md`（評論 B+→A 的具體補法、A 級章節檢查表、
> 各章現況、建議先把 Ch1 深耕成範本）。Quarto 在 `~/.local/quarto/bin/quarto`（非系統 PATH）。
>
> ✅ **已修（測試隔離）**：`make verify` 以前用 demo 資料覆寫 `artifacts/tokenizer.json`（81 字），
> 害有真實中文 ckpt 在場時 serve 單元測試 KeyError。現在 verify 把 demo 產物導到 `artifacts/_verify/`
> （`--artifacts` flag、gitignored），絕不碰真 `artifacts/`；serve 測試 prompt 也改成從 tokenizer 自己
> 的 vocab 取字（不綁特定字元）。**make verify 11/11、make test 39 綠，且不再覆寫真 tokenizer。**
> 中文 tokenizer 重建指令（若日後又被別的操作清掉）：`python pipeline/01_prepare_data.py
> --input data/raw/zhwiki.txt --doc_sep '<|doc|>' --tokenizer char --test_frac 0.1`（確定性、vocab 14210）。

# Session 接續點

斷 session 後，接續者（人或 Claude）先讀：本檔 → `README.md` → `CLAUDE.md` → `docs/lessons-learned.md`。
repo 是自足的真相來源；本檔只給「我們走到哪、下一步可以幹嘛」。

## 一句話現況

從零手刻的小型中文 GPT，已走完**整個 LLM + MLOps 生命週期**：原理 → 現代架構 → 真實中文資料工程
→ 訓練評估 → 部署/可觀測/GPU 容器/Grafana/治理 → 進階 MLOps(e1–e5) → 真候選上線 → **後訓練全弧
（SFT → DPO → IPO → RLHF：GRPO + PPO）**。公開於 GitHub（CI 綠、**59 測試**、全可重跑）。
**並已寫成一本 A 級、MLOps 為主軸的書（`book/`，線上版已部署最新）**；第二本（邊緣運算）已立案、待定範圍。

## 已完成（大圖）

1. **資料**：collect→clean→去重(字元 n-gram+MinHash+LSH)→tokenize→pack；品質偵測器報表；105MB 中文維基實戰
2. **模型**：decoder GPT + 現代零件全可切（RMSNorm/SwiGLU/RoPE/GQA/FlashAttention/KV-cache）
3. **訓練/評估**：先講判準再驗證；BPC、test set、多 seed
4. **部署**：FastAPI(`serve/app.py`)、Prometheus+Grafana(`monitoring/`)、Podman GPU 容器(`Containerfile`)、模型治理(`src/registry.py`)
5. **進階 e1–e5**：漂移監控、重訓迴圈(回歸 gate)、金絲雀/A-B、動態批次、量化壓縮
6. **真候選上線**：8000 步候選 test_loss 3.46 < 舊 3.70，過 gate→promote→**production = digest `4d694be9342d`**，舊 `00b47fc84755` archived
7. **後訓練 SFT**：`pipeline/05_sft.py`，base→會應答格式；正規評估(`scripts/eval_sft.py`)暴露兩個評估陷阱（見 lessons-learned）
8. **後訓練 DPO**：`pipeline/06_dpo.py`，SFT→偏好對齊（policy+凍結 reference，免 reward model）。兩軸對照證明「容量內類推 vs 超出只背」；`eval_dpo` 出 `dpo_generalization.png`
9. **後訓練 RLHF**：reward model（`07`）+ GRPO（`08`）。把 DPO 收合的零件拆開；reward hacking 對照（β=0 RM 暴漲但 mode collapse）；`eval_grpo` 出 `grpo_reward_hacking.png`

## 目前狀態（具體）

- **production 模型** = `artifacts/ckpt.pt`（digest `4d694be9342d`，8000 步，test_loss ~3.46）。registry 台帳：`make models`
- **SFT 模型** = `artifacts/sft_ckpt.pt`（chat 版，沒 promote；評估方式不同）
- **DPO 模型** = `artifacts/dpo_format_ckpt.pt`（會類推那顆）、`artifacts/dpo_ckpt.pt`（topic 對照、只背）；都沒 promote（對齊版，評估尺＝偏好類推率）
- **RLHF 模型** = `artifacts/reward_ckpt.pt`（RM）、`artifacts/grpo_ckpt.pt`（KL 錨穩定版）、`artifacts/grpo_hack_ckpt.pt`（reward-hacked 教學反例）；都沒 promote
- **registry/**（進 git，審計軌跡）：production + archived 兩顆 + cards
- artifacts/（gitignored）：ckpt、bin、語料、sft/dpo 的 jsonl + dpo_*_ckpt.pt + dpo_generalization.png 等
- 監控 stack 可能還開著：`make dashboard-down` 收

## 下一步選項（未拍板，Ryan 挑）

- **🔥 第二本書（邊緣運算）範圍**：Ryan 已立案、在想範圍——全四部 vs 先做第四部「邊緣機隊治理」（建議）。
  拍板後起 `book2-edge/` 骨架。先讀 `book2-edge/IMPROVEMENT-PLAN.md`。
- **第一本書**：A 級主體已完成、已 push、線上版最新；真要再提價值＝「真實規模故事/對外更強 baseline」（非再加輔助）。
- **後訓練再往下**：DPO 多模板 / mask-prompt 精修 / GAE（PPO 的多步優勢估計）——但對齊家族（DPO/IPO/GRPO/PPO）核心已收齊
- **規模**：換更大語料/更大模型（唯一真讓能力上世代的槓桿，但燒算力；DPO topic 軸學不動就是規模牆）
- **k8s**：把容器真部上 k8s-lab（/health→probe、/metrics→ServiceMonitor）——偏 k8s 練習
- 或就此收尾沉澱

## 怎麼接續（關鍵指令）

```bash
cd ~/Documents/llm-from-scratch && source .venv/bin/activate   # uv venv, Python 3.12
make help            # 所有指令
make test            # 回歸（~35 測試）
make models          # 看 registry 台帳（誰是 production）
make serve           # 起推論 API（/docs 試）
make dashboard       # Prometheus+Grafana（放 artifacts/candidate.pt 自動開金絲雀+shadow）
make sft / eval-sft  # 後訓練 SFT + 正規評估
make dpo / eval-dpo  # 後訓練 DPO 偏好對齊 + 類推 vs 死背曲線圖
make reward / grpo / eval-grpo  # 後訓練 RLHF：reward model + GRPO + reward hacking 對照圖
make compare A= B=   # 比兩模型 test_loss + 一致率
```

## 重要心法（別忘）

見 `docs/lessons-learned.md`。最關鍵：**評估要選對指標、質疑指標有沒有被汙染**——同一顆模型、不同的尺、相反的結論（SFT 評估那兩個陷阱就是活例）。
