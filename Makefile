# Makefile = 最樸素的 pipeline orchestrator（之後可換 DVC / Airflow）。
# Java 類比：這就是把四個 Spring Batch step 串起來的 job 定義。
# 每個 target 是一個階段，依賴關係讓 make 自動決定要重跑哪些。

PY := python
ART := artifacts
INPUT := data/raw/input.txt

.PHONY: all data data-demo test book-smoke book-figures verify stats quality serve image run-container dashboard dashboard-down register models retrain compress compare sft-data sft eval-sft dpo-data dpo eval-dpo dpo-beta dpo-ipo reward grpo eval-grpo ppo eval-ppo lab train eval gen plot-loss attn bpe clean smoke help

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

# 書本 book/examples/*.py 的端到端煙霧測試：BOOK_SMOKE=1 縮到極小設定，CPU 幾十秒，
# 只驗「每支範例都跑得動、不崩」。數字正確性的快測在 tests/test_book_examples.py（make test）。
book-smoke: $(INPUT)
	ln -sf ../../$(INPUT) book/examples/input.txt
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_kvcache.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_dedup.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_eval.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_observability.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_serve.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_drift.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_dpo.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_gpt.py
	cd book/examples && BOOK_SMOKE=1 $(PY) tiny_modern.py

# 從真跑數據重新產生書裡的插圖（存進 book/images/）；純 CPU 約 5 分鐘
book-figures: $(INPUT)
	ln -sf ../../$(INPUT) book/examples/input.txt
	cd book/examples && $(PY) make_book_figures.py

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

image:  ## podman 建推論服務 image
	podman build -t llm-from-scratch:latest -f Containerfile .

run-container:  ## GPU 跑容器，模型 mount 進去（需先設好 CDI）
	podman run --rm --device nvidia.com/gpu=all -p 8000:8000 \
	  -v ./artifacts:/app/artifacts:ro,Z llm-from-scratch:latest

dashboard:  ## 起 Prometheus+Grafana 監控 stack（http://127.0.0.1:3000）
	bash monitoring/up.sh

dashboard-down:  ## 收掉監控 stack
	bash monitoring/down.sh

register:  ## 把目前 artifacts 的模型註冊進 registry
	$(PY) scripts/registry_cli.py register

models:  ## 看 model registry 台帳
	$(PY) scripts/registry_cli.py list

retrain:  ## 重訓迴圈（資料→訓練→評估→註冊→gate）；--auto-promote 過 gate 自動上線
	$(PY) scripts/retrain.py --skip-data

compress:  ## 量化壓縮對比（fp32 vs fp16 vs int8：大小 vs 品質）
	$(PY) scripts/compress.py

compare:  ## 比較兩個模型的產出（test_loss + greedy 一致率）：make compare A=ckpt1 B=ckpt2
	$(PY) scripts/compare_models.py $(A) $(B)

sft-data:  ## 從語料自抽 SFT 指令資料（問答 JSONL）
	$(PY) scripts/make_sft_data.py

sft:  ## SFT 指令微調（後訓練里程碑1）：base → 會聽話的對話格式
	$(PY) pipeline/05_sft.py --iters 1500

eval-sft:  ## SFT 專用評估（held-out：回答段 perplexity + 應答行為率）
	$(PY) scripts/eval_sft.py

dpo-data:  ## 自抽 DPO 偏好對（兩種軸：topic=張冠李戴難、format=連貫vs退化易）
	$(PY) scripts/make_dpo_data.py --mode topic
	$(PY) scripts/make_dpo_data.py --mode format \
		--out artifacts/dpo_format.jsonl --heldout artifacts/dpo_format_heldout.jsonl

dpo:  ## DPO 偏好對齊（後訓練里程碑2）：SFT → 偏好較好回答；--beta 控 KL 約束
	$(PY) pipeline/06_dpo.py --iters 600 --beta 0.1 \
		--dpo_data artifacts/dpo_format.jsonl --heldout artifacts/dpo_format_heldout.jsonl \
		--out artifacts/dpo_format_ckpt.pt --log_csv artifacts/runs/dpo_format.csv
	$(PY) pipeline/06_dpo.py --iters 600 --beta 0.1

eval-dpo:  ## DPO 評估（held-out 偏好類推率：format 會類推 vs topic 只死背 + 曲線圖）
	$(PY) scripts/eval_dpo.py

dpo-beta:  ## DPO 精修：掃 β（KL 旋鈕）→ 偏好/漂移/生成重複率三取捨 + 圖
	$(PY) scripts/dpo_beta_sweep.py

reward:  ## RLHF①：訓 reward model（Bradley-Terry，從 SFT 接骨幹 + 純量 head）
	$(PY) pipeline/07_reward_model.py --iters 400

grpo:  ## RLHF②：GRPO 用 RM 分數做 RL（有 KL 錨 + 無 KL 錨對照，揭露 reward hacking）
	$(PY) pipeline/08_grpo.py --iters 150 --beta 0.05 --out artifacts/grpo_ckpt.pt --log_csv artifacts/runs/grpo.csv
	$(PY) pipeline/08_grpo.py --iters 250 --beta 0.0 --lr 1e-4 --out artifacts/grpo_hack_ckpt.pt --log_csv artifacts/runs/grpo_hack.csv

eval-grpo:  ## RLHF 評估：代理(RM)漲 vs 真實(多樣性)崩 = reward hacking 對照圖
	$(PY) scripts/eval_grpo.py

ppo:  ## RLHF（PPO 版）：critic + clipped 目標；clip vs 無 clip 對照（β=0 隔離 clip 的作用）
	$(PY) pipeline/09_ppo.py --iters 120 --clip_eps 0.2 --beta 0 --lr 1e-4 --epochs 8 --out artifacts/ppo_ckpt.pt --log_csv artifacts/runs/ppo_clip.csv
	$(PY) pipeline/09_ppo.py --iters 120 --clip_eps 0 --beta 0 --lr 1e-4 --epochs 8 --out artifacts/ppo_noclip_ckpt.pt --log_csv artifacts/runs/ppo_noclip.csv

eval-ppo:  ## PPO 評估：clip→KL 受控 vs 無 clip→失控 + critic value_loss 下降，出圖
	$(PY) scripts/eval_ppo.py

dpo-ipo:  ## IPO 對照：DPO（margin 爆衝）vs IPO（釘在目標 1/2β、防過度優化）+ 圖
	$(PY) pipeline/06_dpo.py --loss dpo --iters 600 --dpo_data artifacts/dpo_format.jsonl --heldout artifacts/dpo_format_heldout.jsonl --out artifacts/dpo_format_ckpt.pt --log_csv artifacts/runs/cmp_dpo.csv
	$(PY) pipeline/06_dpo.py --loss ipo --iters 600 --dpo_data artifacts/dpo_format.jsonl --heldout artifacts/dpo_format_heldout.jsonl --out artifacts/ipo_format_ckpt.pt --log_csv artifacts/runs/cmp_ipo.csv
	$(PY) scripts/dpo_vs_ipo.py
