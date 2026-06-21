---
title: Session 接續點（斷 session 後從這裡繼續）
type: handoff
updated: 2026-06-21
---

> **最新（2026-06-21 下午）**：完成**後訓練里程碑2＝DPO 偏好對齊**（`pipeline/06_dpo.py`、
> `make dpo`/`eval-dpo`、`tests/test_dpo.py` 5 條，全套 `make test` 39 條綠）。核心發現：
> 兩種偏好軸對照 → format（連貫 vs 退化）held-out **69%→97% 真類推**；topic（對題 vs 張冠李戴）
> 8M **學不動只背 train**（held-out ~9%）。圖 `artifacts/dpo_generalization.png`。
> **DPO 精修＝β 旋鈕掃描**（`make dpo-beta`、`scripts/dpo_beta_sweep.py`）：固定步數下「β 越小漂移越大
> （margin 目標≈1/β）」翻掉教科書直覺；行為兌現＝生成重複率 SFT 8%→DPO 5–6%。圖 `dpo_beta_sweep.png`。
> **後訓練里程碑3＝RLHF（reward model + GRPO）**：`src/reward_model.py`、`pipeline/07_reward_model.py`(`make
> reward`，BT 損失、held-out 偏好 100%)、`pipeline/08_grpo.py`(`make grpo`，GRPO=取樣→RM 打分→組內 advantage→
> PG+KL 錨，不需 critic)、`scripts/eval_grpo.py`(`make eval-grpo`)、`tests/test_rlhf.py` 5 條。**核心對照＝
> reward hacking**：無 KL 錨(β=0) RM 分數 3.7→13.2 暴漲但生成多樣性 100%→6% mode collapse（三題全吐「方言，
> 的方言…」鑽 RM 分布外盲點）＝Goodhart；KL 錨(β>0) 防之。圖 `grpo_reward_hacking.png`。`make test` 44 綠。
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
（SFT → DPO → RLHF）**。公開於 GitHub（CI 綠、44 測試、全可重跑）。

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

- **後訓練再往下**：① IPO（專治 DPO/RLHF 的過度優化）② PPO（補完 RL 家族，比 GRPO 重、有 critic）③ DPO 多模板
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
