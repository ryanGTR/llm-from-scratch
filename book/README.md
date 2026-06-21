# 書：《從零打造並對齊一個 LLM》

📖 **線上閱讀**：<https://ryangtr.github.io/llm-from-scratch/>

用 [Quarto](https://quarto.org) 排版的書（O'Reilly 風：serif 內文、紅色強調、callout 旁註、
程式碼註解、數學渲染）。輸出 HTML（已發佈 GitHub Pages）與 PDF（需 xelatex）。

## 重新部署到 GitHub Pages

```bash
cd book
quarto render --to html                          # 只渲染 HTML（PDF 需 TeX）
# 用 worktree 把 _book/ 推到 gh-pages 分支：
cd .. && git worktree add /tmp/ghp gh-pages
find /tmp/ghp -mindepth 1 -not -path '*/.git*' -delete
cp -r book/_book/. /tmp/ghp/ && touch /tmp/ghp/.nojekyll
cd /tmp/ghp && git add -A && git commit -m "deploy book" && git push origin gh-pages
cd - && git worktree remove /tmp/ghp
```

## 安裝 Quarto

```bash
# Arch（AUR）
yay -S quarto-cli-bin        # 或 paru -S quarto-cli-bin
quarto --version
```

PDF 還需要 LaTeX（CJK 用 xelatex + Noto CJK 字型，Ryan 機器已有 Noto Serif/Sans CJK TC）：

```bash
quarto install tinytex       # 或系統裝 texlive-xetex + texlive-langchinese
```

## 預覽 / 渲染

```bash
cd book
quarto preview               # 本機即時預覽（改檔自動 reload）
quarto render                # 產出 _book/（HTML）
quarto render --to pdf       # 產出 PDF
```

## 結構

```
book/
├── _quarto.yml        # 書本設定（章節、格式、O'Reilly 風）
├── theme.scss         # serif + 紅色強調的樣式
├── index.qmd          # 前言
├── 01..08-*.qmd       # 各章（part 分三部 + 附錄）
├── 99-references.qmd  # 參考文獻
├── references.bib     # 書目
└── images/            # 圖（從 artifacts/ 複製進來，committed）
```

## 內容來源

各章從 repo 既有文件重組而來：`README.md` 發現 1–12、`docs/case-study.md`、`docs/derivations.md`、
`docs/reading-the-charts.md`、`docs/lessons-learned.md`、`docs/theory-map.md`。

## 完成度

骨架 + 前言 + 第 1 章（最小 GPT）+ 第 7 章（對齊，皇冠）+ 第 8 章（數學附錄）內容較完整；
第 2–6 章是「有血有肉的 stub」（開頭 + 關鍵內容 + 圖 + 帶走什麼），標了草稿處待展開成流暢敘事。
把它寫滿是個長期 side project——但骨幹、實驗、數學、圖都已就位。

## 發佈到 GitHub Pages（可選）

```bash
quarto publish gh-pages      # 在 book/ 目錄執行
```
