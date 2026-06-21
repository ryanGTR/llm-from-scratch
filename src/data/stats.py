"""Data quality metrics —「好資料」到底怎麼量？這裡把它變成可計算的數字。

訓 LLM 時「資料好不好」不是感覺，是幾個能算出來的指標：
  - 量夠不夠     total_chars / total_tokens（太少 -> 模型只會背、不會學）
  - 多不多元     char_entropy（資訊量）、compression_ratio（越壓越扁 = 越重複）
  - 乾不乾淨     dedup 率、quality 過濾率（從 pipeline 報表來）
  - 形狀健不健康 文件長度分布、字元頻率分布

每個指標都附一個「粗略健康門檻」+ 判讀（healthy/warn/bad）。門檻是經驗值、
不是鐵律——重點是讓你「有數字可看、有依據可判斷」，而不是憑感覺說資料好。

純計算、不畫圖（畫圖在 notebook）。只用 stdlib + numpy(選用)。
"""

import gzip
import json
import math
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ── 單篇/全文層級的指標 ───────────────────────────────────────────────

def char_entropy(text: str) -> float:
    """Shannon entropy（bits/字元）。越高 = 用字越多元、資訊越豐富。

    直覺：全部都是同一個字 -> 0 bits；字元分布越平均 -> 越接近 log2(vocab)。
    自然英文大約 4.0–4.5 bits/char。太低通常代表重複或退化。
    """
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def compression_ratio(text: str) -> float:
    """gzip 壓縮後大小 / 原始大小。越小 = 越好壓 = 越重複（資訊密度低）。

    自然語言大約 0.3–0.45。逼近 0 代表大量重複；逼近 1 代表像亂數。
    這是「重複度」最便宜好用的代理指標。
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(gzip.compress(raw, compresslevel=6)) / len(raw)


def char_frequencies(text: str, top: int = 20) -> list[tuple[str, int]]:
    """最常見的前 top 個字元（畫長條圖用）。"""
    return Counter(text).most_common(top)


def doc_length_stats(lengths: list[int]) -> dict:
    """文件長度的分布摘要。太多超短文件 = 來源品質可疑。"""
    if not lengths:
        return {}
    s = sorted(lengths)
    n = len(s)
    return {
        "count": n,
        "min": s[0],
        "max": s[-1],
        "mean": round(sum(s) / n, 1),
        "median": s[n // 2],
    }


# ── 整條 pipeline 的彙總指標（從 artifacts 讀）────────────────────────

@dataclass
class Verdict:
    label: str          # 指標名
    value: float | int | str
    status: str         # "healthy" | "warn" | "bad"
    note: str           # 怎麼判讀


def _band(value, healthy_lo, healthy_hi, label, fmt="{:.2f}", note=""):
    if healthy_lo <= value <= healthy_hi:
        st = "healthy"
    elif value < healthy_lo * 0.5 or value > healthy_hi * 1.5:
        st = "bad"
    else:
        st = "warn"
    return Verdict(label, fmt.format(value), st, note)


def pipeline_metrics(artifacts_dir: str | Path = "artifacts") -> dict:
    """讀 artifacts/ 算出整份資料品質指標 + 判讀。notebook 與 CLI 都用這個。"""
    art = Path(artifacts_dir)
    meta = json.loads((art / "meta.json").read_text())
    report = json.loads((art / "data_report.json").read_text())
    corpus = (art / "clean_corpus.txt").read_text(encoding="utf-8")

    H = char_entropy(corpus)
    ratio = compression_ratio(corpus)

    # 從報表算「來源品質」：被過濾/去重的比例
    docs_in = meta["docs_in"]
    drop_quality = next((s["dropped"] for s in report["stages"] if s["stage"] == "quality"), 0)
    drop_exact = next((s["dropped"] for s in report["stages"] if s["stage"] == "exact_dedup"), 0)
    drop_near = next((s["dropped"] for s in report["stages"] if s["stage"] == "near_dedup"), 0)
    dup_rate = (drop_exact + drop_near) / docs_in if docs_in else 0.0
    junk_rate = drop_quality / docs_in if docs_in else 0.0

    metrics = {
        "total_chars": len(corpus),
        "total_tokens": meta["train_tokens"] + meta["val_tokens"] + meta.get("test_tokens", 0),
        "vocab_size": meta["vocab_size"],
        "char_entropy": round(H, 3),
        "entropy_efficiency": round(H / math.log2(meta["vocab_size"]), 3) if meta["vocab_size"] > 1 else 0,
        "compression_ratio": round(ratio, 3),
        "docs_in": docs_in,
        "docs_out": meta["docs_out"],
        "dup_rate": round(dup_rate, 3),
        "junk_rate": round(junk_rate, 3),
        "top_chars": char_frequencies(corpus, 20),
    }

    verdicts = [
        _band(metrics["total_chars"], 1_000_000, 1e12, "資料量(字元)",
              fmt="{:.0f}", note="小型 LM 建議 ≥ 1M 字元；太少只會過擬合背答案"),
        _band(metrics["entropy_efficiency"], 0.55, 0.97, "熵效率(熵/log2 vocab)",
              note="跨語言指標：用字多不多元。英文≈0.78、中文≈0.70；太低=重複/退化"),
        _band(metrics["compression_ratio"], 0.30, 0.70, "壓縮比",
              note="越小越重複；自然語言約 0.3–0.45"),
        _band(metrics["dup_rate"], 0.0, 0.30, "重複率",
              note="去重砍掉的比例；偏高代表來源大量轉貼"),
    ]
    metrics["verdicts"] = [v.__dict__ for v in verdicts]
    return metrics


def print_report(artifacts_dir: str | Path = "artifacts") -> dict:
    """CLI 友善版：印一份文字報表（make stats 用），回傳 metrics。"""
    m = pipeline_metrics(artifacts_dir)
    icon = {"healthy": "✅", "warn": "⚠️ ", "bad": "❌"}
    print("=" * 56)
    print("資料品質報表")
    print("=" * 56)
    print(f"  文件：{m['docs_in']} -> {m['docs_out']}    "
          f"字元：{m['total_chars']:,}    token：{m['total_tokens']:,}")
    print(f"  vocab_size：{m['vocab_size']}    "
          f"字元熵：{m['char_entropy']} bits    壓縮比：{m['compression_ratio']}")
    print("-" * 56)
    for v in m["verdicts"]:
        print(f"  {icon[v['status']]} {v['label']:16s} {str(v['value']):>10}   {v['note']}")
    print("=" * 56)
    return m


if __name__ == "__main__":
    import sys
    print_report(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
