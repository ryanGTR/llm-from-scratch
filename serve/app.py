"""推論 API（FastAPI）：把訓練好的 GPT 包成線上服務。

訓練是離線一次性；服務是線上持續——要快（KV-cache）、要能被呼叫、要看得到狀態。
模型在啟動時載入一次（不是每個請求都載），之後常駐記憶體服務請求。

用法：make serve → 開 http://127.0.0.1:8000/docs 試，或
  curl -s localhost:8000/generate -H 'content-type: application/json' \
       -d '{"prompt":"數學是","max_new_tokens":80}'

Java 類比：lifespan 載入 = Spring 的 @PostConstruct 單例 Bean；@app.post = @RestController。
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig          # noqa: E402
from src.model import GPT                  # noqa: E402
from src.tokenizer import load_tokenizer   # noqa: E402

ART = Path(os.environ.get("ARTIFACTS", "artifacts"))
STATE: dict = {}


def _load_model():
    """啟動時載入一次：ckpt + tokenizer 常駐記憶體。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ART / "ckpt.pt", map_location=device)
    cfg = GPTConfig(**ckpt["gpt_config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    STATE.update(
        model=model, cfg=cfg, device=device,
        tok=load_tokenizer(ART / "tokenizer.json"),
        params_M=round(model.num_params() / 1e6, 3),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()        # 服務起來前先把模型載好
    yield


app = FastAPI(title="llm-from-scratch 推論 API", version="0.1.0", lifespan=lifespan)


class GenRequest(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = 0.9
    min_p: float | None = None
    use_kv_cache: bool = True


class GenResponse(BaseModel):
    text: str
    prompt: str
    generated_tokens: int
    latency_ms: float
    tokens_per_sec: float


@app.get("/health")
def health():
    """就緒探針：模型載好沒、跑在哪、多大。給 k8s readiness probe 用。"""
    ok = "model" in STATE
    return {
        "status": "ok" if ok else "loading",
        "model_loaded": ok,
        "device": STATE.get("device"),
        "params_M": STATE.get("params_M"),
        "vocab_size": STATE["cfg"].vocab_size if ok else None,
    }


@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest):
    """自回歸生成。回傳生成文字 + 延遲/吞吐（線上服務要看得到效能）。"""
    model, tok, device = STATE["model"], STATE["tok"], STATE["device"]
    ids = tok.encode(req.prompt) or [0]      # 空/全生字 → 用一個起始 token 兜底
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            idx, req.max_new_tokens, temperature=req.temperature,
            top_k=req.top_k, top_p=req.top_p, min_p=req.min_p,
            use_kv_cache=req.use_kv_cache,
        )
    if device == "cuda":
        torch.cuda.synchronize()             # 等 GPU 真的算完再計時
    dt = time.perf_counter() - t0

    full = out[0].tolist()
    n_gen = len(full) - len(ids)
    return GenResponse(
        text=tok.decode(full),
        prompt=req.prompt,
        generated_tokens=n_gen,
        latency_ms=round(dt * 1000, 1),
        tokens_per_sec=round(n_gen / dt, 1) if dt > 0 else 0.0,
    )
