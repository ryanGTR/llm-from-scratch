"""Model Registry CLI：註冊 / 列出 / 升 production。

  python scripts/registry_cli.py register        # 把目前 artifacts 的模型註冊
  python scripts/registry_cli.py list            # 看台帳
  python scripts/registry_cli.py promote <short> # 升 production（要過 gate）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import registry as R   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register")
    sub.add_parser("list")
    p = sub.add_parser("promote")
    p.add_argument("short")
    args = ap.parse_args()

    if args.cmd == "register":
        e = R.register()
        print(f"已註冊 {e['short']}（狀態 {e['status']}）-> registry/cards/{e['short']}.md")
    elif args.cmd == "list":
        rows = R.load_registry()
        if not rows:
            print("（台帳是空的，先 register）")
        for e in rows:
            m, lin = e["metrics"], e["lineage"]
            flag = "🟢" if e["status"] == "production" else "  "
            print(f"{flag} {e['short']}  {e['status']:11s} "
                  f"test_loss={m.get('test_loss')} "
                  f"quality_gate={lin.get('data_quality_gate')} "
                  f"commit={lin.get('code_commit')}")
    elif args.cmd == "promote":
        ok, msg = R.promote(args.short)
        print(("✅ " if ok else "❌ ") + msg)


if __name__ == "__main__":
    main()
