---
title: 第二本書（邊緣運算）計畫（斷 session 可接續）
type: plan
created: 2026-06-22
updated: 2026-06-22
tags: [book2, edge, slm, on-device, mlops, plan, handoff]
---

# 第二本書：把 LLM 推到邊緣（計畫，尚未起骨架）

> **接續者先讀這頁。** 第二本書＝邊緣運算，是第一本（`book/`，從零刻 LLM→上線/治理/維護）的續集。
> **2026-06-22 決定方向＋前置已拍板，但「範圍」Ryan 還在想——所以目前只記錄、還沒建任何章節/骨架。**

## ⏩ 目前狀態
- **已決定**：寫第二本書，主題＝**邊緣運算 / on-device + 邊緣機隊治理**。接第一本續集（同一顆模型當載體）。
- **前置已拍板（2026-06-22）**：
  1. **放哪**：同 repo，新增 `book2-edge/`（共用第一本的模型/治理程式碼）。←本檔就在這。
  2. **真跑硬體**：Ryan **沒有 ARM 裝置**。→ edge 部署實驗在 **FW16 CPU + llama.cpp / ONNX** 上做，
     **誠實標注「裝置為 CPU 模擬、非真實 ARM/NPU」**。鐵則同第一本：嵌入數字都要真跑、不杜撰。
  3. **範圍**：⏳ **未定，Ryan「先想想」**。我的建議＝**先做第四部（邊緣機隊治理）**——最接他 moat
     （受監管產業治理）、且 CPU 可完整模擬真跑、CP 值最高。等他拍板才起骨架。

## 為什麼這本（定位）
- 第一本：從零刻 LLM → 養到上線/治理/維護（雲/server 的 MLOps）。
- 第二本：把那顆模型**推到資源受限的邊緣裝置**，並治理**一整隊**邊緣裝置。
- 價值：①補第一本誠實缺口（沒真實裝置部署）②雙倍下注 moat（受監管產業的邊緣機隊治理：分行設備/
  醫療/工控）③少人寫的交集（from-scratch 理解 × edge 部署 × 治理）。

## 提議大綱（四部；沿用第一本 A-級配方）
1. **為什麼上邊緣**：離線/隱私/延遲/成本/**受監管資料不出境** + edge 約束（記憶體/算力/功耗/CPU-NPU）。
2. **讓模型變小變快**：量化 int8/int4、剪枝、蒸餾；KV-cache/取樣在 edge regime；真實 size/latency/品質三角。
3. **真的部到 edge**：匯出 ONNX 或 llama.cpp/GGUF，在 CPU（模擬裝置）上跑與量測、離線推論。
4. **邊緣機隊治理（moat）**：N 台裝置的 digest/版本、分批 OTA 更新、fleet 上的 canary/shadow、
   每裝置 drift→在地重訓 or 召回、可稽核回滾。受監管產業視角。

**每章 A-級配方（沿用第一本，見 `../book/IMPROVEMENT-PLAN.md`）**：①學習目標框（可摺疊）
②Mermaid「你在這裡」地圖 ③「🎯 給技術主管」術語速查框（可摺疊）④一段逐行走讀
⑤💻 純 CPU 真跑範例（數字不杜撰）⑥章末 3 題（預測/動手/弄壞，答案摺疊）。
工具：Quarto（`~/.local/quarto/bin/quarto`）、Mermaid（原生、CJK OK）、繁中內文、術語保留英文。

## 第一部可用的市場素材（2026-06-22 已整理，可當第一部底稿）
SLM 代表作（Phi-3/4、Llama 3.2 1B/3B、Gemma 2/3+3n、Qwen2.5 小檔、Apple ~3B on-device、Ministral、
SmolLM2、BitNet 1-bit；大廠小檔 GPT-4o-mini/Haiku/Gemini Nano）。明確應用：裝置端助理、agentic 工作馬
（NVIDIA 2025〈SLMs are the Future of Agentic AI〉）、私有 RAG、edge/IoT/車載、領域專用小模型、
高量窄任務換成本、隱私/受監管。市場觀點：「small is the new big」、企業 ROI 在 SLM、right-size 到任務、
on-device 成硬體賣點（AI PC/NPU）、Ollama/llama.cpp/GGUF 民主化；反面＝SLM 非 AGI、贏在窄/專用/成本/隱私。
**可選**：要引用來源/最新數字的版本 → 跑 deep-research skill 出一份可查證報告。

## 下一步
- 等 Ryan 拍板**範圍**（全四部 vs 先第四部）。拍板後：起 `book2-edge/` 骨架（`_quarto.yml` + 章節 +
  `examples/`）、把本檔的「⏩ 目前狀態」更新成進度格（同第一本慣例）。
- 在那之前**不動手建章節**。第一本一路本機 commit、未 push，維持不動。
