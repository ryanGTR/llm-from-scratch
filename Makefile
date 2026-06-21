# Makefile = 最樸素的 pipeline orchestrator（之後可換 DVC / Airflow）。
# Java 類比：這就是把四個 Spring Batch step 串起來的 job 定義。
# 每個 target 是一個階段，依賴關係讓 make 自動決定要重跑哪些。

PY := python
ART := artifacts
INPUT := data/raw/input.txt

.PHONY: all data data-demo test verify stats quality serve lab train eval gen plot-loss attn bpe clean smoke help

help:
	@echo "make data      - 下載樣本語料並跑資料 pipeline"
	@echo "make data-demo - 產一份髒語料，示範清洗/去重各砍了什麼"
	@echo "make train  - 訓練（FW16 上請用: prime-run make train）"
	@echo "make eval   - 算 val loss / perplexity"
	@echo "make gen    - 讓模型續寫一段"
	@echo "make plot-loss - 把訓練 loss 畫成曲線（artifacts/loss_curve.png，多 run 比較）"
	@echo "make all    - data -> train -> eval -> gen 全跑"
	@echo "make smoke  - 極小設定快速跑通整條 pipeline（驗證用）"
	@echo "make test   - 跑資料 pipeline 的自動測試（驗證它真的有效）"
	@echo "make verify - 跑驗證 playbook：逐項印 PASS/FAIL（驗收用）"
	@echo "make stats  - 印資料品質報表（量/熵/壓縮比/重複率 + 健康判讀）"
	@echo "make lab    - 開 Jupyter Lab（探索 notebook，需先 source .venv）"
	@echo "make clean  - 清掉 artifacts"

all: gen

$(INPUT):
	bash scripts/get_data.sh

data: $(INPUT)
	$(PY) pipeline/01_prepare_data.py --input $(INPUT)

# 示範用：故意很髒的小語料，跑完看 artifacts/data_report.json
data-demo:
	$(PY) scripts/make_messy_corpus.py
	$(PY) pipeline/01_prepare_data.py --input data/raw/demo

$(ART)/ckpt.pt: data
	$(PY) pipeline/02_train.py

train: $(ART)/ckpt.pt

eval: train
	$(PY) pipeline/03_eval.py

gen: train
	$(PY) pipeline/04_generate.py --prompt "\n" --max_new_tokens 300

plot-loss:
	$(PY) pipeline/plot_loss.py

# 極小設定：CPU 幾分鐘內跑完，用來確認整條鏈沒壞
smoke: data
	$(PY) pipeline/02_train.py --max_iters 200 --block_size 32 --n_layer 2 --n_embd 64
	$(PY) pipeline/03_eval.py --eval_iters 20
	$(PY) pipeline/04_generate.py --max_new_tokens 200

test:
	$(PY) -m unittest discover -s tests -v

# 驗證 playbook：操作者導向的驗收，逐項 PASS/FAIL，全過才 exit 0
verify:
	$(PY) scripts/verify.py

# 資料品質報表（量/多元性/重複/乾淨度 + 健康判讀）
stats:
	$(PY) -m src.data.stats $(ART)

# 開 Jupyter Lab 玩探索 notebook（MATPLOTLIBRC 讓圖表中文不變方框）
lab:
	MATPLOTLIBRC=$(CURDIR)/matplotlibrc jupyter lab notebooks/

clean:
	rm -f $(ART)/*.bin $(ART)/*.json $(ART)/*.pt $(ART)/*.png $(ART)/*.txt
	rm -rf $(ART)/runs

attn:
	$(PY) pipeline/viz_attention.py

bpe:
	$(PY) scripts/train_bpe.py --merges 500

quality:
	$(PY) scripts/quality_report.py --doc_sep "<|doc|>"

serve:  ## 起推論 API（http://127.0.0.1:8000/docs）
	$(PY) -m uvicorn serve.app:app --host 127.0.0.1 --port 8000 --reload
