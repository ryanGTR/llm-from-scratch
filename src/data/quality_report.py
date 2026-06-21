"""資料品質偵測器 + 報表：把「看資料」工業化成可審的報表。

心法：你看不完 11k 篇。所以「看樣本發現問題 → 把它寫成一條偵測器 → 偵測器從此
自動掃全語料 + 未來每批新資料 → 再看還有什麼怪 → 再寫一條」。偵測器累積起來
就是「資料品質即程式碼」（資料的測試套件）。每條設門檻，超標就 fail = 資料 gate。

Java 類比：每個 Detector 就像一條 Bean Validation 規則，quality_report 是對整批
資料跑驗證、彙總成一份可稽核的報告。
"""

import re
from dataclasses import dataclass
from typing import Callable

from .clean import _max_char_ratio, _symbol_ratio

_WIKI = re.compile(r"-\{[^}]*\}-")              # MediaWiki 繁簡/語言轉換語法 -{...}-
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class Detector:
    name: str
    test: Callable[[str], bool]   # (text) -> True 代表「有問題」
    threshold_pct: float          # 命中比例超過此 % 就算 fail（gate）
    note: str
    pattern: "re.Pattern | None" = None   # 有的話，樣本顯示「命中片段+前後文」


def _pat(p):
    """包一個 regex 偵測器：test = 有沒有命中。"""
    return p


# 偵測器清單——發現新問題就在這裡加一條（累積成資料的測試套件）
DETECTORS = [
    Detector("wiki_markup", lambda t: bool(_WIKI.search(t)), 0.5, "維基 -{…}- 語言轉換語法殘留", _WIKI),
    Detector("html_residue", lambda t: bool(_HTML.search(t)), 0.5, "HTML 標籤殘留", _HTML),
    Detector("control_chars", lambda t: bool(_CTRL.search(t)), 0.1, "控制字元", _CTRL),
    Detector("url", lambda t: bool(_URL.search(t)), 5.0, "含 URL", _URL),
    Detector("symbol_spam", lambda t: _symbol_ratio(t) > 0.3, 1.0, "符號佔比過高"),
    Detector("high_repetition", lambda t: _max_char_ratio(t) > 0.3, 1.0, "單一字元洗版"),
    Detector("too_short", lambda t: len(t.strip()) < 50, 5.0, "過短文件(<50 字)"),
]


def _snippet(text: str, d: Detector) -> str:
    """報表樣本：regex 偵測器顯示「命中片段 + 前後 25 字」，其他顯示文件開頭。"""
    if d.pattern is not None:
        m = d.pattern.search(text)
        if m:
            s, e = max(0, m.start() - 25), min(len(text), m.end() + 25)
            return ("…" + text[s:e] + "…").replace("\n", " ")
    return text.strip().replace("\n", " ")[:80]


def quality_report(docs: list[str], detectors=DETECTORS, sample_n: int = 3) -> dict:
    """對整批文件跑所有偵測器，回傳可審報表：每項命中數/%/過不過/問題樣本。"""
    n = len(docs) or 1
    rows = []
    for d in detectors:
        hits = [i for i, t in enumerate(docs) if d.test(t)]
        pct = len(hits) / n * 100
        rows.append({
            "name": d.name,
            "hits": len(hits),
            "pct": round(pct, 2),
            "threshold_pct": d.threshold_pct,
            "pass": pct <= d.threshold_pct,
            "note": d.note,
            "samples": [_snippet(docs[i], d) for i in hits[:sample_n]],
        })
    return {
        "total_docs": len(docs),
        "all_pass": all(r["pass"] for r in rows),   # 全過才算這批資料「乾淨到可用」
        "detectors": rows,
    }
