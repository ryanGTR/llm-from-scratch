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

- [ ] 🟢 樸素 attention → **FlashAttention**（torch `F.scaled_dot_product_attention`，一行換掉）→ context 上限實測可 ×4
- [ ] 🔴 生成加 **KV-cache**：每步只算新 token，O(T²)→O(T)（真實 LLM 推論標配）
- [ ] 🟢 取樣 top-k → **top-p（nucleus）**（現代主流取樣）
- 對應缺陷：技術債「沒接 Flash/KV-cache」「取樣是舊的」。

## 3. tokenizer 補強

- [ ] 🟡 BPE 從 char-level → **byte-level**（處理任何 unicode/emoji，修「遇生字直接丟掉」的真 bug）
- [ ] 🟡 BPE 效率：純 Python O(N×merges) 太慢 → 增量計數，或直接接 **tiktoken**
- [ ] 🟢 `.bin` uint16 → **uint32**（解 vocab 上限卡在 65535）
- 對應缺陷：技術債「BPE 慢/char 起步」「uint16 上限」。

## 4. 實驗嚴謹度（讓數據站得住腳）

- [ ] 🟡 每個實驗**多 seed 重跑**，給平均 + 誤差線（現在單次跑，128vs256 差 0.026 在雜訊內）
- [ ] 🟢 加獨立 **test set**（現在只有 train/val）
- 對應缺陷：「實驗不嚴謹、README 略高估」。做完才能把「趨勢」變「定論」。

## 5. 工程化（往生產級靠）

- [ ] 🟡 config 管理：argparse → **Hydra / OmegaConf**（或 pydantic-settings），超參數集中可組合
- [ ] 🟡 實驗追蹤：**Weights & Biases (W&B)** 或 MLflow，自動記每次 run 的參數+曲線
- [ ] 🟢 依賴鎖定：**`uv lock`**（uv.lock）保證換機器可重現
- [ ] 🟡 **CI（GitHub Actions）**：push 自動跑 `make test`
- [ ] 🟡 資料/模型版本：**DVC**
- 對應缺陷：「無 config/CI/lockfile/實驗追蹤」。

## 6. 能力：放大（需要算力，不只是技巧）

- [ ] 🔴 放大模型/資料：上**雲端 GPU**（runpod / Lambda / vast.ai），pipeline 不用改、只換 device
- [ ] 🔴 post-training：**SFT（指令微調）→ DPO**（Direct Preference Optimization，比 RLHF 簡單的現代對齊）→ 把「接龍機器」變「會聽話的助理」
- 對應缺陷：「本質是玩具/沒 post-training」。

## 7. 學習：換我主導（內化，不是看）

- [ ] 🟡 上面挑一塊，**Ryan 親手寫/debug，Claude 只在旁糾錯**（補「我懂≈我看懂」這個缺陷）

---

## 建議起手順序

1. **FlashAttention（2-①）**🟢 — 一行換掉、立刻看到 context 上限 ×4，回報最高
2. **多 seed 重跑（4-①）**🟡 — 讓既有數據站得住，也順便練實驗紀律
3. **現代零件 RMSNorm/SwiGLU（1）**🟡 — 把骨架升級成現代款，最有「跟上最新」的感覺
4. 之後再往 KV-cache / RoPE / 工程化 / post-training 推進

> 每做一項：先預測效果 → 改 → 看 `02_monitor.ipynb` 的曲線 → 跑 `make test` 守住。
