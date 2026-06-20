"""Central config — all hyper-parameters live here.

Java 類比：這就是你的 application.yml / @ConfigurationProperties。
四個 pipeline 階段都讀同一份設定，確保「同一組超參數」可重現。
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # 模型結構（先用會跑得動的小尺寸；學原理夠用）
    block_size: int = 128      # context length：一次最多看幾個 token
    n_layer: int = 4           # Transformer block 疊幾層
    n_head: int = 4            # multi-head attention 的頭數
    n_embd: int = 128          # embedding 維度（必須能被 n_head 整除）
    dropout: float = 0.1
    bias: bool = False         # Linear/LayerNorm 要不要 bias
    vocab_size: int = 0        # 由 tokenizer 決定，prepare_data 後寫入
    # 現代化開關（可切換對比 classic vs modern）
    use_rmsnorm: bool = False  # True = RMSNorm（LLaMA 同款）；False = LayerNorm
    use_swiglu: bool = False   # True = SwiGLU MLP（LLaMA 同款）；False = GELU MLP
    use_rope: bool = False      # True = RoPE 旋轉位置編碼；False = 學習式 position embedding


@dataclass
class TrainConfig:
    # 訓練流程
    batch_size: int = 32
    max_iters: int = 2000
    eval_interval: int = 200   # 每幾步算一次 val loss
    eval_iters: int = 50       # 算 loss 時取樣幾個 batch
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337

    # 路徑（pipeline 各階段的產物落點）
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


# 兩份預設值，腳本可以 import 後覆寫
DEFAULT_GPT = GPTConfig()
DEFAULT_TRAIN = TrainConfig()
