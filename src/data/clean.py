"""Clean 階段：正規化文字 + 用「品質規則」過濾整篇爛文件。

兩件事分清楚：
  1. normalize_text  —— 把一篇文字「洗乾淨」（修，不丟）。
  2. quality_check   —— 判斷整篇要不要「丟掉」（垃圾文件直接扔）。

Garbage in, garbage out。模型不會比你的資料更聰明，這層就是把關。
Java 類比：normalize = 你的 DTO 正規化；quality_check = Bean Validation。
"""

import re
import unicodedata
from dataclasses import dataclass

# 預先編譯的 regex（效能 + 可讀性）
_HTML_TAG = re.compile(r"<[^>]+>")
_MANY_SPACES = re.compile(r"[ \t]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
# 控制字元（除了 \n \t）——這些常是亂碼來源
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(s: str, strip_html: bool = True) -> str:
    """把單篇文字洗乾淨：unicode 正規化、去 HTML、去控制字元、收斂空白。"""
    s = unicodedata.normalize("NFC", s)        # 全形/組合字統一表示法
    if strip_html:
        s = _HTML_TAG.sub(" ", s)
    s = _CTRL.sub("", s)                        # 砍掉控制字元
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _MANY_SPACES.sub(" ", s)               # 多個空白 -> 一個
    s = _MANY_NEWLINES.sub("\n\n", s)          # 3+ 換行 -> 段落間距
    # 每行去尾端空白
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()


@dataclass
class QualityConfig:
    min_chars: int = 20            # 太短的文件資訊量低
    max_symbol_ratio: float = 0.3  # 非「字母/數字/空白/中日韓」佔比上限
    max_repeat_ratio: float = 0.3  # 單一字元佔比上限（擋 "aaaaaa..." 這種）


def _symbol_ratio(s: str) -> float:
    if not s:
        return 1.0
    def is_content(c: str) -> bool:
        if c.isalnum() or c.isspace():
            return True
        # 中日韓統一表意文字也算「內容」，不算符號
        return "一" <= c <= "鿿"
    symbols = sum(1 for c in s if not is_content(c))
    return symbols / len(s)


def _max_char_ratio(s: str) -> float:
    if not s:
        return 1.0
    counts: dict[str, int] = {}
    for c in s:
        if not c.isspace():
            counts[c] = counts.get(c, 0) + 1
    if not counts:
        return 1.0
    return max(counts.values()) / sum(counts.values())


def quality_check(s: str, cfg: QualityConfig) -> tuple[bool, str]:
    """回傳 (是否保留, 原因)。原因用來印報表，讓你看到「為什麼被丟」。"""
    if len(s) < cfg.min_chars:
        return False, "too_short"
    if _symbol_ratio(s) > cfg.max_symbol_ratio:
        return False, "too_many_symbols"
    if _max_char_ratio(s) > cfg.max_repeat_ratio:
        return False, "too_repetitive"
    return True, "ok"
