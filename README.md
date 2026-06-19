# llm-from-scratch

從零手刻一個小型 GPT，並把「資料 → 訓練 → 評估 → 生成」做成一條可重跑的
pipeline。目標是**搞懂 LLM 原理**（A），同時練習**模型生產流水線的工程化**（C）。
從本機（FW16 + NVIDIA 5070）起步，結構保留得很乾淨，之後可平滑搬上雲端 GPU。

> 這不是要做 ChatGPT。這裡的模型是 ~0.1–10M 參數的 char-level GPT，
> 小到能在你自己機器上一個下午訓完，但麻雀雖小五臟俱全——
> attention、residual、自回歸生成、訓練迴圈，全都是真的。

## 心智模型（Java 類比）

| 這個專案 | Java 世界 |
|---|---|
| `src/model.py` 的訓練/前向邏輯 | 你的 business logic（一支 `main()`）|
| `Makefile` 串四個階段 | Spring Batch job：step1→step2→… |
| `artifacts/ckpt.pt` 權重檔 | 編譯產物 `.jar` |
| `src/tokenizer.py` | serializer，String ↔ int[] |
| `pipeline/03_eval.py` | CI 的測試 gate（品質夠才放行）|

## 結構

```
llm-from-scratch/
├── src/                  # 核心：模型 + tokenizer + config
│   ├── config.py         #   所有超參數（= application.yml）
│   ├── tokenizer.py      #   char-level tokenizer
│   └── model.py          #   minimal GPT（decoder-only Transformer）
├── src/data/             # 資料子系統（純 Python、零依賴）
│   ├── sources.py        #   collect：來源 -> Document 清單
│   ├── clean.py          #   normalize + 品質過濾
│   └── dedup.py          #   exact + near-dup（MinHash）
├── pipeline/             # 四個階段，每個是一支可獨立跑的腳本
│   ├── 01_prepare_data.py  # collect→clean→dedup→tokenize→pack，含報表
│   ├── 02_train.py
│   ├── 03_eval.py
│   └── 04_generate.py
├── scripts/get_data.sh   # 下載樣本語料
├── Makefile              # pipeline orchestrator
├── data/raw/             # 原始 .txt 放這
└── artifacts/            # 產物：tokenizer / *.bin / ckpt.pt / 報告
```

## 環境需求

- Python（建議 **3.12**，見下方「踩雷」）
- PyTorch（CUDA 版才吃得到 NVIDIA 5070）、NumPy
- FW16：跑 GPU 要用 `prime-run` 包起來

### 環境隔離：用 uv（已採用）

系統 Python 是 3.14，PyTorch 官方 wheel 還沒支援，所以本專案用 **uv** 釘一個
獨立的 **Python 3.12** 環境（不碰系統 Python）。uv = 「管 Python 版本 + 管套件」
一把抓，類比 Java 的 SDKMAN!（管 JDK 版本）+ Maven（管依賴）合體。

```bash
cd ~/Documents/llm-from-scratch
uv venv --python 3.12 .venv     # 建立隔離環境（uv 自動抓一份 3.12）
source .venv/bin/activate       # 進入環境；提示字首會出現 (.venv)
uv pip install numpy jupyterlab # 資料/探索用；torch 下面單獨裝

# 之後要訓練再裝 CUDA 版 torch（~2GB+，對應機器 CUDA，cu124 常見）：
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

> 每次要跑都先 `source .venv/bin/activate`。離開用 `deactivate`。
> 純資料 pipeline（make test / verify / data-demo）只需 stdlib，免裝也能跑。

## 跑跑看

```bash
# 資料這塊（不需要 torch，現在就能跑）：看清洗/去重各砍了什麼
make data-demo
cat artifacts/data_report.json      # 完整報表
make stats                          # 資料品質：量/熵/壓縮比/重複率 + 健康判讀
make verify                         # 驗收 playbook：逐項 PASS/FAIL
# 細節見 docs/data-pipeline.md

# 探索資料（A 學原理線，需 source .venv）：互動算指標 + 畫圖
make lab                            # 開 Jupyter Lab -> notebooks/01_explore_data.ipynb

# 0) 先確認整條鏈沒壞（極小設定，CPU 幾分鐘）
make smoke

# 監控（B）：把 loss 存 CSV 畫成曲線，比較不同設定
python pipeline/02_train.py --max_iters 3000 --run_name big        # 命名這次訓練
python pipeline/02_train.py --max_iters 3000 --n_embd 64 --run_name small
make plot-loss                      # 疊圖比較 -> artifacts/loss_curve.png

# 1) 正式流程
make data                  # 下載語料 + tokenize
prime-run make train       # 用 NVIDIA GPU 訓練
make eval                  # 看 val loss / perplexity
make gen                   # 讓模型續寫

# 想換自己的文本：把任意 .txt 覆蓋到 data/raw/input.txt 再 make data
```

訓練初期 loss 從 ~4.x（= ln(vocab)，等於亂猜）往下掉就對了；
char-level shakespeare 訓到 val loss ~1.5 左右，生成的文字會開始有英文的「形狀」。

## Roadmap（學習路線）

- **階段 0｜先跑通**：`make smoke` 跑完整條 pipeline，確認環境 OK。
- **階段 1｜懂原理**：讀 `src/model.py`，對照 Karpathy「Let's build GPT」影片，
  逐行搞懂 attention / mask / loss。改 `config.py` 參數觀察 loss 變化。
- **階段 2｜工程化**：把超參數抽成 `configs/*.yaml`；用 DVC 追蹤資料與模型版本；
  把 `03_eval` 變成有門檻的 gate（loss 沒過就不算 pass）。
- **階段 3｜換 tokenizer**：char-level → BPE（`tiktoken` 或自己刻），體會 subword。
- **階段 4｜上雲**：把同一條 pipeline 跑到雲端 GPU（runpod / Lambda / vast.ai），
  程式碼不用改，只換 device 與資料路徑——這就是 pipeline 化的回報。
- **階段 5｜轉 B 路線**：拿開源權重（Qwen/Llama）做 LoRA 微調，做出能用的模型。

## See Also

- `docs/theory-map.md` — 程式碼 ↔ 聖經本章節 / 論文 對照地圖
- `docs/data-pipeline.md`、`docs/verification-playbook.md`
- Karpathy, *Let's build GPT: from scratch*（YouTube）
- Karpathy, **nanoGPT**（本專案的精神來源）
