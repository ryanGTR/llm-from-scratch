#!/usr/bin/env bash
# 收掉整個監控 stack。
podman pod rm -f llm-stack 2>/dev/null && echo "llm-stack 已收掉" || echo "沒有在跑的 llm-stack"
