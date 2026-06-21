"""推論 API（FastAPI）：把訓練好的 GPT 包成線上服務。

訓練是離線一次性；服務是線上持續——要快（KV-cache）、要能被呼叫、要看得到狀態。
模型在啟動時載入一次（不是每個請求都載），之後常駐記憶體服務請求。

用法：make serve → 開 http://127.0.0.1:8000/docs 試，或
  curl -s localhost:8000/generate -H 'content-type: application/json' \
       -d '{"prompt":"數學是","max_new_tokens":80}'

Java 類比：lifespan 載入 = Spring 的 @PostConstruct 單例 Bean；@app.post = @RestController。
"""

import json
import logging
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import GPTConfig          # noqa: E402
from src.drift import DriftMonitor         # noqa: E402
from src.model import GPT                  # noqa: E402
from src.registry import load_registry, sha256_file   # noqa: E402
from src.tokenizer import load_tokenizer   # noqa: E402

ART = Path(os.environ.get("ARTIFACTS", "artifacts"))
# 金絲雀：設了 CANDIDATE_CKPT 就同時載入候選模型，CANARY_PCT% 的流量導去它
CANDIDATE_CKPT = os.environ.get("CANDIDATE_CKPT")
CANARY_PCT = float(os.environ.get("CANARY_PCT", "0"))
STATE: dict = {}

# ---- 可觀測性：Prometheus metrics（k8s/Grafana 同款標準）----------------------
# Counter 只增（總量）；Histogram 自動分桶（看 p50/p95 延遲分布）。
REQS = Counter("llm_requests_total", "請求總數", ["endpoint", "status"])
LATENCY = Histogram("llm_request_latency_seconds", "請求延遲(秒)", ["endpoint"])
TOKENS = Counter("llm_generated_tokens_total", "累計生成 token 數")
DRIFT_PSI = Gauge("llm_drift_psi", "請求字元分布 vs 訓練分布的 PSI（>0.25 顯著漂移）")
DRIFT_OOV = Gauge("llm_drift_oov_rate", "請求用到訓練外字元的比例")
VARIANT = Counter("llm_variant_requests_total", "各版本接到的請求數", ["variant"])
VARIANT_LAT = Histogram("llm_variant_latency_seconds", "各版本延遲", ["variant"])

# 結構化日誌（每請求一行 JSON → 生產環境送到 log 聚合器）
logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("serve")


def log_event(**kw):
    _log.info(json.dumps({"ts": round(time.time(), 3), **kw}, ensure_ascii=False))


def _load_model():
    """啟動時載入一次：ckpt + tokenizer 常駐記憶體（含可選的金絲雀候選模型）。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_one(path):
        ck = torch.load(path, map_location=device)
        m = GPT(GPTConfig(**ck["gpt_config"])).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        return m, f"sha256:{sha256_file(path)}"

    model, digest = load_one(ART / "ckpt.pt")
    STATE.update(
        model=model, cfg=GPTConfig(**torch.load(ART / "ckpt.pt", map_location=device)["gpt_config"]),
        device=device,
        tok=load_tokenizer(ART / "tokenizer.json"),
        params_M=round(model.num_params() / 1e6, 3),
        digest=digest,                                # production 模型身份
        drift=DriftMonitor.from_artifacts(ART),
        candidate=None, candidate_digest=None,
    )
    if CANDIDATE_CKPT and Path(CANDIDATE_CKPT).exists():
        cand, cand_digest = load_one(Path(CANDIDATE_CKPT))
        STATE["candidate"] = cand
        STATE["candidate_digest"] = cand_digest       # 金絲雀候選


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()        # 服務起來前先把模型載好
    yield


app = FastAPI(title="llm-from-scratch 推論 API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    """對「每個」請求自動量延遲、計數、記狀態——不用每個 handler 自己寫。"""
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - t0
    ep = request.url.path
    REQS.labels(ep, response.status_code).inc()
    LATENCY.labels(ep).observe(dt)
    return response


@app.get("/drift")
def drift_info():
    """漂移報告：線上請求 vs 訓練分布偏離多少。retrain_suggested=True → 該重訓了。"""
    return STATE["drift"].report()


@app.get("/model")
def model_info():
    """治理閉環：回報「我正在服務哪個 digest」+ 它在 registry 的狀態。

    審計用：問 live API → 拿 digest → 對台帳 → 確認線上是不是被批准的 production 那顆。
    若 status=UNREGISTERED，代表上線了一顆「沒登記」的模型 = 紅旗。
    """
    digest = STATE.get("digest")
    match = next((e for e in load_registry() if e["model_digest"] == digest), None)
    return {
        "serving_digest": digest,
        "in_registry": match is not None,
        "status": match["status"] if match else "UNREGISTERED",
        "metrics": match["metrics"] if match else None,
        "data_quality_gate": match["lineage"].get("data_quality_gate") if match else None,
        "canary": ({"digest": STATE["candidate_digest"], "traffic_pct": CANARY_PCT}
                   if STATE.get("candidate") is not None else None),
    }


@app.get("/metrics")
def metrics():
    """Prometheus 抓取端點：請求數/延遲分布/token 總量。給 Grafana 畫儀表板。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    variant: str = "production"        # production / canary（金絲雀）
    served_digest: str | None = None


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
    tok, device = STATE["tok"], STATE["device"]
    drift = STATE["drift"]
    drift.observe(req.prompt)                # 餵給漂移監控（累計線上分布）
    DRIFT_PSI.set(drift.psi())
    DRIFT_OOV.set(drift.oov_rate())

    # 金絲雀路由：有候選 + 擲骰子落在 CANARY_PCT% → 走 canary，否則 production
    if STATE.get("candidate") is not None and random.random() * 100 < CANARY_PCT:
        model, variant, served = STATE["candidate"], "canary", STATE["candidate_digest"]
    else:
        model, variant, served = STATE["model"], "production", STATE["digest"]
    VARIANT.labels(variant).inc()

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
    VARIANT_LAT.labels(variant).observe(dt)  # 各版本延遲分開記 → 才能 A/B 比較

    full = out[0].tolist()
    n_gen = len(full) - len(ids)
    TOKENS.inc(n_gen)                        # 累計生成 token（Prometheus）
    log_event(event="generate", variant=variant, prompt_len=len(ids), tokens=n_gen,
              latency_ms=round(dt * 1000, 1), kv_cache=req.use_kv_cache)  # 結構化日誌
    return GenResponse(
        text=tok.decode(full),
        prompt=req.prompt,
        generated_tokens=n_gen,
        latency_ms=round(dt * 1000, 1),
        tokens_per_sec=round(n_gen / dt, 1) if dt > 0 else 0.0,
        variant=variant,
        served_digest=served,
    )
