#!/usr/bin/env bash
# 下載一份小語料當練習素材：tiny-shakespeare（~1MB 純文字）。
# 換成你自己的 .txt 也可以——丟到 data/raw/input.txt 即可。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
echo "下載 tiny-shakespeare ..."
curl -fsSL "$URL" -o data/raw/input.txt
wc -c data/raw/input.txt
