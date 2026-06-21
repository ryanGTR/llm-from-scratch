# CLAUDE.md — llm-from-scratch

給 Claude 的工作說明。專案目的：從零手刻小型 GPT，並做成可重跑的
data→train→eval→generate pipeline。學原理（A）+ 工程化（C），本機起步、可上雲。

## 慣例

- 程式碼註解可用中文輔助說明，但**核心術語保留英文**。
- 每個 pipeline 階段 = `pipeline/0N_*.py`，可獨立執行，產物落 `artifacts/`。
- 超參數集中在 `src/config.py`，腳本以 argparse 覆寫，不要散落硬編碼。
- 新增階段要同步更新 `Makefile` 與 `README.md` 的結構圖。

## 跑法

- 快速驗證整條鏈：`make smoke`（極小設定，CPU 可跑）。
- FW16 上吃 GPU：訓練/生成前面加 `prime-run`。
- 改動模型後，至少跑 `make smoke` 確認沒壞再回報。

## 環境

- 用 **uv** + 專案內 `.venv`（**Python 3.12.13**）做隔離；系統 Python 3.14 無 torch wheel。
- 跑任何需要套件的東西前先 `source .venv/bin/activate`。
- 已裝：numpy、jupyterlab、**torch（cu128，已驗 RTX 5070 sm_120 可用）**、datasets（抓語料）。
- 資料 pipeline（make test / verify / data-demo / quality）需 **numpy**（去重 MinHash+LSH 用 numpy 向量化），不需 torch。

## 已知狀況

- artifacts/、data/raw/、.venv/ 已在 .gitignore，不要 commit。

## 風格

- 遵循全域 `~/.claude/CLAUDE.md`：繁中回答、先結論後細節、改檔前先讀。
