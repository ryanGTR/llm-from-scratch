---
title: 資料 Pipeline（collect → clean → dedup → tokenize → pack）
type: note
created: 2026-06-19
updated: 2026-06-19
tags: [llm, data, pipeline, etl]
sources: [src/data/, pipeline/01_prepare_data.py]
---

# 資料 Pipeline

訓練 LLM 約 **80% 的工夫在資料**，不在模型。模型架構大家都抄 Transformer，
真正拉開差距的是「餵什麼、餵得多乾淨」。這份文件解釋本專案的資料這一棒。

> Java 類比：整條就是一個 **ETL job**。collect = ItemReader，
> clean/dedup = ItemProcessor，pack = ItemWriter。`01_prepare_data.py` 是 job 定義。

## 一張圖

```
data/raw/  ──collect──▶ [Document, Document, ...]
                          │ clean      每篇洗乾淨（normalize：去 HTML/控制字元/收斂空白）
                          │ quality    丟掉爛文件（太短/符號太多/太重複）
                          │ exact dedup 內容雜湊相同 → 丟（O(n)）
                          │ near dedup  MinHash 估相似度 ≥ 門檻 → 丟（近似轉貼）
                          ▼
                       concat ──tokenize──▶ token ids ──split──▶ train/val ──pack──▶ *.bin
```

每一步都印「處理前/後」數字，並寫進 `artifacts/data_report.json`。

## 各階段

| 階段 | 檔案 | 在做什麼 | 為什麼重要 |
|---|---|---|---|
| collect | `src/data/sources.py` | 把來源讀成一篇篇 `Document` | 以「文件」為單位，去重/過濾才有意義 |
| clean | `src/data/clean.py` `normalize_text` | unicode 正規化、去 HTML、去控制字元、收斂空白 | 亂碼會變成 vocab 裡的垃圾 token |
| quality | `src/data/clean.py` `quality_check` | 丟太短/符號太多/單字元洗版的文件 | 低品質文件拉低模型上限 |
| exact dedup | `src/data/dedup.py` `exact_dedup` | 內容雜湊（SHA-1）相同就丟 | 重複資料 = 模型死背、浪費算力 |
| near dedup | `src/data/dedup.py` `near_dedup` | MinHash 簽章估 Jaccard 相似度 | 抓「改幾個字的轉貼」，exact 抓不到 |
| pack | `pipeline/01_prepare_data.py` | tokenize → split → 寫 uint16 `.bin` | 訓練階段 mmap 讀取要的格式 |

## MinHash 一句話原理

把每篇切成「k 個詞一組」的碎片集合，用 N 個雜湊函數各取碎片的最小雜湊值，
得到一條長度 N 的簽章。**兩條簽章在每一格相等的機率，剛好等於兩集合的
Jaccard 相似度**——所以比簽章就能估相似度，不必兩兩硬比原文。

本專案用「貪婪兩兩比簽章」O(n²)，小語料夠用。大規模要加 **LSH** 把候選對
壓到接近 O(n)——這正是之後拿 **Rust 重寫練系統程式**的好題目。

## 動手玩

```bash
make data-demo                 # 產髒語料 + 跑 pipeline，看每關砍了什麼
cat artifacts/data_report.json # 完整報表（哪些被丟、為什麼）
less artifacts/clean_corpus.txt # 去重後的乾淨全文，肉眼檢查

# 對照組：關掉去重看 token 數差多少
python pipeline/01_prepare_data.py --input data/raw/demo --no_dedup
```

## 下一步可做（階段2 工程化）

- [ ] quality 門檻抽到 config／加語言偵測（只留中/英）
- [ ] near dedup 換成 MinHash + LSH，量大才不會 O(n²) 爆掉
- [ ] 加資料版本控管（DVC），讓「這個模型用哪版資料」可追溯
- [ ] tokenizer 從 char-level 換成 BPE，看 vocab 與壓縮率怎麼變

## See Also

- `README.md` — 專案總覽與 roadmap
- `src/data/dedup.py` — MinHash 實作（含註解）
