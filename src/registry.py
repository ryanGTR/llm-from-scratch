"""Model Registry：模型治理的核心。

治理＝對「線上的模型」答得出四個稽核問題：哪一個(身份)、吃什麼訓的(lineage)、
表現如何(metrics/card)、憑什麼上線(promotion gate)。

身份用 ckpt 的 sha256 digest（像 container image digest / cosign，不可變、可驗證），
不靠檔名。registry.json 是可審計的單據台帳；每筆綁 lineage + 指標 + git commit + 狀態。

Java 類比：registry.json 像 Maven repository 的 metadata，但加上「來源可追溯 + 上線簽核」。
"""

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REG_DIR = Path("registry")
REG_FILE = REG_DIR / "registry.json"
CARD_DIR = REG_DIR / "cards"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_registry() -> list[dict]:
    if REG_FILE.exists():
        return json.loads(REG_FILE.read_text())
    return []


def _save_registry(reg: list[dict]):
    REG_DIR.mkdir(parents=True, exist_ok=True)
    REG_FILE.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def build_entry(art: Path) -> dict:
    """從 artifacts 蒐集一筆 registry 紀錄（不寫檔，純組裝 → 可測）。"""
    import torch
    ckpt_path = art / "ckpt.pt"
    digest = sha256_file(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = json.loads((art / "meta.json").read_text())

    evalr = {}
    if (art / "eval_report.json").exists():
        evalr = json.loads((art / "eval_report.json").read_text())
    dq = {}
    if (art / "data_quality_report.json").exists():
        dq = json.loads((art / "data_quality_report.json").read_text())
    data_digest = (f"sha256:{sha256_file(art / 'train.bin')[:16]}"
                   if (art / "train.bin").exists() else None)

    return {
        "model_digest": f"sha256:{digest}",
        "short": digest[:12],
        "status": "registered",                       # 預設未上線；要 promote 才 production
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": ckpt.get("gpt_config", {}),
        "params_M": evalr.get("params_M"),
        "metrics": {k: evalr.get(k) for k in ("val_loss", "test_loss", "perplexity")
                    if evalr.get(k) is not None},
        "lineage": {
            "data_digest": data_digest,               # 資料的不可變身份
            "corpus_docs": meta.get("docs_out"),
            "vocab_size": meta.get("vocab_size"),
            "train_tokens": meta.get("train_tokens"),
            "tokenizer": meta.get("tokenizer"),
            "data_quality_gate": dq.get("all_pass"),   # 資料品質 gate 過了沒
            "code_commit": _git_commit(),              # 程式碼版本
        },
    }


def register(art="artifacts") -> dict:
    """把目前 artifacts 的模型註冊進台帳（同 digest 不重複），並產出 model card。"""
    art = Path(art)
    entry = build_entry(art)
    reg = load_registry()
    if any(e["model_digest"] == entry["model_digest"] for e in reg):
        return next(e for e in reg if e["model_digest"] == entry["model_digest"])
    reg.append(entry)
    _save_registry(reg)
    write_card(entry)
    return entry


def gate_reasons(entry: dict, current_prod: dict | None = None,
                 tol: float = 0.05) -> list[str]:
    """promotion gate 規則（純函式 → 可測）：回傳「擋下的理由」，空 list = 可上線。

    current_prod 有給的話多一道「回歸檢查」：新模型 test_loss 不能比現行 production 差太多
    （超過 tol）。自動重訓必備——免得迴圈把一顆更爛的模型推上線。
    """
    reasons = []
    if entry.get("lineage", {}).get("data_quality_gate") is not True:
        reasons.append("資料品質 gate 未通過")
    new_loss = entry.get("metrics", {}).get("test_loss")
    if new_loss is None:
        reasons.append("缺 test 評估")
    if current_prod and new_loss is not None:
        prod_loss = current_prod.get("metrics", {}).get("test_loss")
        if prod_loss is not None and new_loss > prod_loss + tol:
            reasons.append(f"回歸：test_loss {new_loss} 比現行 production {prod_loss} 差")
    return reasons


def promote(short: str) -> tuple[bool, str]:
    """升 production——但要先過 gate（資料品質 + 有 test 評估）。其餘同時降級。"""
    reg = load_registry()
    match = [e for e in reg if e["short"].startswith(short)]
    if not match:
        return False, f"找不到 {short}"
    entry = match[0]
    current_prod = next((e for e in reg if e.get("status") == "production"
                         and e["model_digest"] != entry["model_digest"]), None)
    gate = gate_reasons(entry, current_prod)
    if gate:
        return False, "promotion gate 擋下：" + "、".join(gate)
    for e in reg:
        if e.get("status") == "production":
            e["status"] = "archived"               # 同時只有一顆 production
    entry["status"] = "production"
    _save_registry(reg)
    write_card(entry)
    return True, f"{entry['short']} 已升 production"


def write_card(entry: dict):
    """產出人讀的 model card（治理單據）。"""
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    c, m, lin = entry["config"], entry["metrics"], entry["lineage"]
    card = f"""# Model Card — `{entry['short']}`

| 欄位 | 值 |
|---|---|
| 模型 digest | `{entry['model_digest']}` |
| 狀態 | **{entry['status']}** |
| 建立時間 | {entry['created_at']} |
| 參數量 | {entry.get('params_M')} M |
| 程式碼 commit | `{lin.get('code_commit')}` |

## 架構
- n_layer={c.get('n_layer')}, n_head={c.get('n_head')}, n_embd={c.get('n_embd')}, block_size={c.get('block_size')}
- RMSNorm={c.get('use_rmsnorm')}, SwiGLU={c.get('use_swiglu')}, RoPE={c.get('use_rope')}, n_kv_head={c.get('n_kv_head')}, Flash={c.get('use_flash')}

## 指標
- val_loss={m.get('val_loss')}, test_loss={m.get('test_loss')}, perplexity={m.get('perplexity')}

## 資料 Lineage（來源可追溯）
- 資料 digest：`{lin.get('data_digest')}`
- 語料文件數：{lin.get('corpus_docs')}、vocab={lin.get('vocab_size')}、train_tokens={lin.get('train_tokens')}
- tokenizer：{lin.get('tokenizer')}
- **資料品質 gate：{'✅ 通過' if lin.get('data_quality_gate') else '❌ 未過 / 未知'}**

## 侷限
- 小模型（char-level、單機規模）：學會字/詞/語法/標點，但整體不連貫，不適合生產內容生成。
- 訓練語料為中文維基，簡繁混合；領域外表現未知。
"""
    (CARD_DIR / f"{entry['short']}.md").write_text(card, encoding="utf-8")
