"""監控：把訓練的 loss 紀錄畫成曲線，並把多次 run 疊在一起比較。

讀 artifacts/runs/*.csv（每次訓練一個檔），畫 train（虛線）/ val（實線）曲線，
存成 artifacts/loss_curve.png。之後你調參數（n_layer/n_embd/iters）跑不同 run，
一張圖就看得出「哪組設定比較好」。

Java 類比：等同把 Spring Boot Actuator 的 metrics 接到 Grafana 看趨勢，
只是這裡輕量到一個 CSV + 一張 PNG。

用法：python pipeline/plot_loss.py        （畫所有 run）
      python pipeline/plot_loss.py default big   （只畫指定幾個 run）
"""

import csv
import sys
from pathlib import Path

# matplotlib 改成「延遲 import」（放在 main 裡）——這樣只用 load_run 解析 CSV
# 的人（例如單元測試）不必裝 matplotlib。CI 抓到的就是這個隱藏依賴。


def load_run(csv_path: Path):
    steps, train, val = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return steps, train, val


def main():
    import matplotlib
    matplotlib.use("Agg")          # 無 GUI 也能存圖（CLI / 遠端都行）
    import matplotlib.pyplot as plt

    art = Path("artifacts")
    runs_dir = art / "runs"
    wanted = sys.argv[1:]   # 指定 run 名；不給就全畫

    csvs = sorted(runs_dir.glob("*.csv"))
    if wanted:
        csvs = [p for p in csvs if p.stem in wanted]
    if not csvs:
        print(f"找不到 loss 紀錄（{runs_dir}/*.csv）。先跑 `make train` 產生。")
        return

    plt.figure(figsize=(9, 5))
    summary = []
    for p in csvs:
        steps, train, val = load_run(p)
        if not steps:
            continue
        line, = plt.plot(steps, val, "-", label=f"{p.stem} val")
        plt.plot(steps, train, "--", color=line.get_color(), alpha=0.5,
                 label=f"{p.stem} train")
        summary.append((p.stem, min(val), val[-1]))

    plt.xlabel("訓練步數 (step)")
    plt.ylabel("loss（越低越好）")
    plt.title("訓練 loss 曲線（實線=val 驗證集，虛線=train 訓練集）")
    plt.legend(); plt.grid(True, alpha=0.3)
    out = art / "loss_curve.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"已存：{out}")
    print("-" * 44)
    print(f"{'run':16s} {'best val':>10} {'final val':>10}")
    for name, best, final in summary:
        print(f"{name:16s} {best:>10.4f} {final:>10.4f}")
    # train 明顯比 val 低很多 = 過擬合的徵兆，提醒一下
    print("\n判讀：兩線越貼近越好；train 遠低於 val = 開始過擬合（資料太少/模型太大）。")


if __name__ == "__main__":
    main()
