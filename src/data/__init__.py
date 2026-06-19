"""資料子系統：collect -> clean -> dedup -> (tokenize/pack 在 pipeline 層).

訓 LLM 約 80% 的工夫在這裡。每個模組都是純 Python、零依賴，
方便你逐行讀懂；之後想拿其中一塊（尤其 dedup）用 Rust 重寫練手也很適合。
"""
