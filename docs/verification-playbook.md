---
title: 驗證 Playbook（資料 Pipeline 驗收 runbook）
type: runbook
created: 2026-06-19
updated: 2026-06-19
tags: [llm, data, verification, playbook, acceptance]
sources: [scripts/verify.py, pipeline/01_prepare_data.py, tests/test_data.py]
---

# 驗證 Playbook — 資料 Pipeline

**用途**：不靠「作者口頭說 OK」，而是任何操作者照這份 runbook 執行，
就能對資料 pipeline 做**驗收**。每一項都有「指令／預期結果／判定」，
全部可重跑、可機器判定。

> 定位（對照你的 IDP 治理思路）：
> - 這份 = **acceptance runbook / playbook**（操作者導向，驗「對外行為」）。
> - `tests/test_data.py` = **開發者回歸測試**（unit，驗「內部單元」）。
> - 兩者互補：playbook 收斂在「驗收標準」，test 守在「改壞就紅燈」。
> - `scripts/verify.py` 就是這份 playbook 的**可執行版**（等同 Ansible 的 assert tasks）。

## 一鍵執行

```bash
make verify          # 跑整份 playbook，逐項印 PASS/FAIL，全過 exit 0
echo $?              # 0 = 全部通過；1 = 有項目失敗（可接 CI / pre-commit）
```

輸出長這樣（節錄）：

```
[PASS] 輸入文件數 == 13  (實際 13)
[PASS] 品質過濾丟掉 3 篇  (實際 3，原因 {'too_repetitive':1,'too_short':1,'too_many_symbols':1})
[PASS] exact dedup 丟掉 1 篇
[PASS] near dedup 丟掉 1 篇
[PASS] 乾淨全文不含 HTML 標籤 '<'
[PASS] .bin round-trip：train 解碼回乾淨全文前段
通過 11/11
```

## 前置條件

| 項目 | 要求 |
|---|---|
| Python | 3.x（純 stdlib，**不需要 torch / numpy**）|
| 工作目錄 | `~/Documents/llm-from-scratch` |
| 網路 | 不需要（用內建的示範髒語料，不對外下載）|

## 驗收項（針對示範語料 `data/raw/demo`）

| # | 驗收項 | 怎麼驗 | 預期 | 判定 |
|---|---|---|---|---|
| 0 | pipeline 能跑完 | 重建語料並執行 `01_prepare_data.py` | exit 0 | 非 0 即 FAIL |
| 1 | 輸入文件數 | 讀 `artifacts/meta.json` `docs_in` | 13 | 不符即 FAIL |
| 2 | 輸出文件數 | `meta.json` `docs_out` | 8 | 不符即 FAIL |
| 3 | 品質過濾砍量 | `data_report.json` quality.dropped | 3 | 不符即 FAIL |
| 4 | 過濾原因涵蓋 | quality.reasons 的 key | 含 短/重複/符號 | 缺一即 FAIL |
| 5 | exact dedup 砍量 | exact_dedup.dropped | 1 | 不符即 FAIL |
| 6 | near dedup 砍量 | near_dedup.dropped | 1 | 不符即 FAIL |
| 7 | 清洗去 HTML | `clean_corpus.txt` 中 `<` 數 | 0 | >0 即 FAIL |
| 8 | 清洗去控制字元 | `clean_corpus.txt` 含 NUL? | 無 | 有即 FAIL |
| 9 | .bin 無損 | 解碼 `train.bin` 比對乾淨全文前段 | 相符 | 不符即 FAIL |
| 10 | 回歸測試 | `tests/` 全部 unittest | 全綠 | 任一紅即 FAIL |

> 驗收數字（13/8/3/1/1）寫在 `scripts/verify.py` 的 `EXPECTED`，
> 與本表一致。改了示範語料就同步改這兩處。

## 手動逐項驗（不想用 verify.py 時）

```bash
make data-demo
python -c "import json;m=json.load(open('artifacts/meta.json'));print(m['docs_in'],m['docs_out'])"  # 13 8
python -c "import json;[print(s['stage'],s['dropped']) for s in json.load(open('artifacts/data_report.json'))['stages']]"
grep -c '<' artifacts/clean_corpus.txt   # 0
```

## 失敗了怎麼辦

1. 看 FAIL 那行的 `-> 實際 ...`，先確認是「程式壞了」還是「示範語料被改過」。
2. 若你**故意**改了 `scripts/make_messy_corpus.py`，同步更新 `EXPECTED` 與上表。
3. near dedup 是機率性（MinHash）；若它在**壓門檻**的案例上飄動，那是預期特性，
   不是 bug——驗收案例要用「明確的近似重複」，別挑壓 0.6 線的邊界文。

## 之後可延伸

- [ ] 把 `make verify` 掛進 git pre-commit / CI，改壞自動擋。
- [ ] 加「資料閘門」：drop 率超過門檻就讓 pipeline 直接中止 + 寫 `metrics.json`。
- [ ] 訓練階段的 playbook：驗 loss 有下降、checkpoint 有產出、生成不是亂碼。

## See Also

- `docs/data-pipeline.md` — 各階段在做什麼
- `scripts/verify.py` — 本 playbook 的執行器
- `tests/test_data.py` — 開發者回歸測試
