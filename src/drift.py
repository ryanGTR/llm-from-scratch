"""Drift 監控：偵測線上請求是否偏離「訓練資料的分布」。

模型不會自己壞掉，是世界變了、它跟不上（concept/data drift）。兩個訊號：
- OOV rate：請求用到「訓練時沒看過的字」的比例。char-level 模型遇到生字只能瞎猜。
- PSI（Population Stability Index）：請求字元分布 vs 訓練分布的偏移量。風控/信評常用——
  PSI < 0.1 穩定、0.1–0.25 輕微漂移、> 0.25 顯著漂移（該考慮重訓了）。

Java 類比：像對「線上流量」持續跑一個健康檢查 monitor，偏離基準就亮燈。
"""

import math
from collections import Counter
from pathlib import Path


class DriftMonitor:
    def __init__(self, baseline: dict[str, float], topk: int = 50):
        self.base = baseline               # char -> 訓練分布比例
        self.vocab = set(baseline)         # 訓練看過的字
        # PSI 要分箱（設計給 ~10-20 桶，不能直接套 1.4 萬字）：取 Top-K 高頻字當桶、其餘併成 "other"
        top = sorted(baseline.items(), key=lambda kv: -kv[1])[:topk]
        self.buckets = [ch for ch, _ in top]
        self.base_b = {ch: baseline[ch] for ch in self.buckets}
        self.base_other = max(1e-6, 1.0 - sum(self.base_b.values()))
        self.obs = Counter()               # 線上累計觀測
        self.n_req = 0
        self.oov_total = 0
        self.char_total = 0

    @classmethod
    def from_corpus(cls, text: str, sample: int = 2_000_000):
        c = Counter(text[:sample])
        tot = sum(c.values()) or 1
        return cls({ch: n / tot for ch, n in c.items()})

    @classmethod
    def from_artifacts(cls, art="artifacts"):
        corpus = (Path(art) / "clean_corpus.txt").read_text(encoding="utf-8")
        return cls.from_corpus(corpus)

    def observe(self, text: str):
        self.n_req += 1
        for ch in text:
            self.obs[ch] += 1
            self.char_total += 1
            if ch not in self.vocab:
                self.oov_total += 1

    def oov_rate(self) -> float:
        return self.oov_total / self.char_total if self.char_total else 0.0

    def psi(self) -> float:
        """分箱 PSI：Top-K 高頻字各一桶 + 其餘併「other」。兩邊加 epsilon 避免 log(0)。"""
        obs_tot = sum(self.obs.values())
        if not obs_tot:
            return 0.0
        psi = 0.0
        obs_other = obs_tot
        for ch in self.buckets:
            b = max(self.base_b[ch], 1e-6)
            o = max(self.obs.get(ch, 0) / obs_tot, 1e-6)
            psi += (o - b) * math.log(o / b)
            obs_other -= self.obs.get(ch, 0)
        o = max(obs_other / obs_tot, 1e-6)        # "other" 桶
        psi += (o - self.base_other) * math.log(o / self.base_other)
        return psi

    def report(self) -> dict:
        psi = self.psi()
        level = "穩定" if psi < 0.1 else ("輕微漂移" if psi < 0.25 else "顯著漂移")
        return {
            "requests": self.n_req,
            "chars_seen": self.char_total,
            "oov_rate": round(self.oov_rate(), 4),
            "psi": round(psi, 4),
            "level": level,
            "retrain_suggested": psi >= 0.25,    # 顯著漂移 → 建議觸發重訓
        }
