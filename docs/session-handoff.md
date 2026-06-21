---
title: Session 接續點（斷 session 後從這裡繼續）
type: handoff
updated: 2026-06-21
---

# Session 接續點

斷 session 後，接續者（人或 Claude）先讀：本檔 → `README.md` → `CLAUDE.md` → `docs/lessons-learned.md`。
repo 是自足的真相來源；本檔只給「我們走到哪、下一步可以幹嘛」。

## 一句話現況

從零手刻的小型中文 GPT，已走完**整個 LLM + MLOps 生命週期**：原理 → 現代架構 → 真實中文資料工程
→ 訓練評估 → 部署/可觀測/GPU 容器/Grafana/治理 → 進階 MLOps(e1–e5) → 真候選上線 → **後訓練 SFT
里程碑（含正規評估）**。公開於 GitHub（CI 綠、~35 測試、全可重跑）。

## 已完成（大圖）

1. **資料**：collect→clean→去重(字元 n-gram+MinHash+LSH)→tokenize→pack；品質偵測器報表；105MB 中文維基實戰
2. **模型**：decoder GPT + 現代零件全可切（RMSNorm/SwiGLU/RoPE/GQA/FlashAttention/KV-cache）
3. **訓練/評估**：先講判準再驗證；BPC、test set、多 seed
4. **部署**：FastAPI(`serve/app.py`)、Prometheus+Grafana(`monitoring/`)、Podman GPU 容器(`Containerfile`)、模型治理(`src/registry.py`)
5. **進階 e1–e5**：漂移監控、重訓迴圈(回歸 gate)、金絲雀/A-B、動態批次、量化壓縮
6. **真候選上線**：8000 步候選 test_loss 3.46 < 舊 3.70，過 gate→promote→**production = digest `4d694be9342d`**，舊 `00b47fc84755` archived
7. **後訓練 SFT**：`pipeline/05_sft.py`，base→會應答格式；正規評估(`scripts/eval_sft.py`)暴露兩個評估陷阱（見 lessons-learned）

## 目前狀態（具體）

- **production 模型** = `artifacts/ckpt.pt`（digest `4d694be9342d`，8000 步，test_loss ~3.46）。registry 台帳：`make models`
- **SFT 模型** = `artifacts/sft_ckpt.pt`（chat 版，沒 promote；評估方式不同）
- **registry/**（進 git，審計軌跡）：production + archived 兩顆 + cards
- artifacts/（gitignored）：ckpt、bin、語料、sft.jsonl、sft_heldout.jsonl 等
- 監控 stack 可能還開著：`make dashboard-down` 收

## 下一步選項（未拍板，Ryan 挑）

- **後訓練往下**：① 把 SFT 做「正規」（mask 指令只算回答段 loss、加結束標記、乾淨停止）② RL/DPO（偏好對齊；DPO 比 PPO/GRPO 簡單，是「會思考」那條的入門）
- **規模**：換更大語料/更大模型（唯一真讓能力上世代的槓桿，但燒算力）
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
make compare A= B=   # 比兩模型 test_loss + 一致率
```

## 重要心法（別忘）

見 `docs/lessons-learned.md`。最關鍵：**評估要選對指標、質疑指標有沒有被汙染**——同一顆模型、不同的尺、相反的結論（SFT 評估那兩個陷阱就是活例）。
