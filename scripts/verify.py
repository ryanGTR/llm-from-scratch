"""驗證 Playbook（執行器）——把「驗收標準」寫成可重跑的程序。

定位：這是操作者導向的「驗收 runbook」，不是開發者的 unit test。
任何人（包含未來的你、CI）跑 `make verify`，就會逐項印出 PASS/FAIL，
最後一行給總結，全過 exit 0、有錯 exit 1。對照 docs/verification-playbook.md。

Java/Ops 類比：等同 Ansible playbook 裡一連串 assert task，
或驗收測試（acceptance test）——宣告「期望狀態」，再檢查真實狀態符不符。
零依賴，純 stdlib。
"""

import json
import subprocess
import sys
import unittest
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))   # 讓 import src.* / tests.* 找得到

# ── 驗收標準（針對示範髒語料 data/raw/demo，與 playbook 文件一致）──────────
EXPECTED = {
    "docs_in": 13,
    "docs_out": 8,
    "quality_dropped": 3,     # 太短1 + 太重複1 + 符號洗版1
    "exact_dropped": 1,
    "near_dropped": 1,
}

results: list[tuple[bool, str, str]] = []   # (pass?, 檢查項, 細節)


def check(name: str, ok: bool, detail: str = ""):
    results.append((ok, name, detail))


def run(cmd: list[str]):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def stage(report: dict, name: str) -> dict:
    for s in report["stages"]:
        if s["stage"] == name:
            return s
    return {}


def main():
    print("=" * 60)
    print("驗證 PLAYBOOK：llm-from-scratch 資料 pipeline")
    print("=" * 60)

    # ── 步驟 0：準備乾淨環境，重跑 pipeline（可重現是前提）──────────────
    print("\n[setup] 重建示範語料並重跑資料 pipeline ...")
    run(["python", "scripts/make_messy_corpus.py"])
    proc = run(["python", "pipeline/01_prepare_data.py", "--input", "data/raw/demo"])
    check("pipeline 正常結束（exit 0）", proc.returncode == 0,
          proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
    if proc.returncode != 0:
        return report_and_exit()

    meta = json.loads((ART / "meta.json").read_text())
    report = json.loads((ART / "data_report.json").read_text())

    # ── 驗收項 1：文件進出數 ──────────────────────────────────────────
    check(f"輸入文件數 == {EXPECTED['docs_in']}",
          meta["docs_in"] == EXPECTED["docs_in"],
          f"實際 {meta['docs_in']}")
    check(f"輸出文件數 == {EXPECTED['docs_out']}",
          meta["docs_out"] == EXPECTED["docs_out"],
          f"實際 {meta['docs_out']}")

    # ── 驗收項 2：各關真的有砍東西 ────────────────────────────────────
    q = stage(report, "quality")
    check(f"品質過濾丟掉 {EXPECTED['quality_dropped']} 篇",
          q.get("dropped") == EXPECTED["quality_dropped"],
          f"實際 {q.get('dropped')}，原因 {q.get('reasons')}")
    check("品質過濾涵蓋 短/重複/符號 三種原因",
          set(q.get("reasons", {})) >= {"too_short", "too_repetitive", "too_many_symbols"},
          f"{q.get('reasons')}")

    ex = stage(report, "exact_dedup")
    check(f"exact dedup 丟掉 {EXPECTED['exact_dropped']} 篇",
          ex.get("dropped") == EXPECTED["exact_dropped"],
          f"實際 {ex.get('dropped')}")

    nr = stage(report, "near_dedup")
    check(f"near dedup 丟掉 {EXPECTED['near_dropped']} 篇",
          nr.get("dropped") == EXPECTED["near_dropped"],
          f"實際 {nr.get('dropped')}")

    # ── 驗收項 3：清洗確實生效（產物裡不該有髒東西）──────────────────
    corpus = (ART / "clean_corpus.txt").read_text(encoding="utf-8")
    check("乾淨全文不含 HTML 標籤 '<'", "<" not in corpus,
          f"找到 {corpus.count('<')} 個")
    check("乾淨全文不含 NUL 控制字元", "\x00" not in corpus, "")

    # ── 驗收項 4：打包的 .bin 能無損 decode 回原文 ───────────────────
    from src.tokenizer import CharTokenizer
    tok = CharTokenizer.load(ART / "tokenizer.json")
    a = array("H")
    a.frombytes((ART / "train.bin").read_bytes())
    decoded = tok.decode(list(a))
    check(".bin round-trip：train 解碼回乾淨全文前段",
          corpus.startswith(decoded),
          f"train={len(a)} tokens")

    # ── 驗收項 5：開發者回歸測試也全綠 ────────────────────────────────
    print("\n[unit] 跑開發者回歸測試 (make test 等價) ...")
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    res = unittest.TextTestRunner(verbosity=0).run(suite)
    check(f"單元測試全過（{res.testsRun} 條）",
          res.wasSuccessful(),
          f"失敗 {len(res.failures)} / 錯誤 {len(res.errors)}")

    report_and_exit()


def report_and_exit():
    print("\n" + "=" * 60)
    print("驗收結果")
    print("=" * 60)
    passed = 0
    for ok, name, detail in results:
        tag = "PASS" if ok else "FAIL"
        line = f"[{tag}] {name}"
        if detail and not ok:
            line += f"  -> {detail}"
        elif detail:
            line += f"  ({detail})"
        print(line)
        passed += ok
    total = len(results)
    print("-" * 60)
    print(f"通過 {passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
