#!/usr/bin/env bash
# 起整個監控 stack：一個 podman pod 裝 API + Prometheus + Grafana（共享 localhost）。
# 這正是「多容器要一起跑、互相發現」——k8s 在解的問題的微縮版（pod 概念也是從這來的）。
set -e
cd "$(dirname "$0")/.."
POD=llm-stack

podman pod rm -f "$POD" 2>/dev/null || true
# port 在 pod 層發布（pod 內容器共享網路）
podman pod create --name "$POD" -p 8000:8000 -p 9090:9090 -p 3000:3000

# 1) 推論 API（GPU + 模型 mount）
podman run -d --pod "$POD" --name llm-api --device nvidia.com/gpu=all \
  -v ./artifacts:/app/artifacts:ro,Z llm-from-scratch:latest

# 2) Prometheus（抓 API 的 /metrics）
podman run -d --pod "$POD" --name llm-prometheus \
  -v ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro,Z \
  docker.io/prom/prometheus:latest

# 3) Grafana（datasource + dashboard 都 provisioning 進去，匿名免登入）
podman run -d --pod "$POD" --name llm-grafana \
  -e GF_AUTH_ANONYMOUS_ENABLED=true \
  -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
  -e GF_AUTH_DISABLE_LOGIN_FORM=true \
  -v ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro,Z \
  -v ./monitoring/grafana/dashboards:/etc/grafana/dashboards:ro,Z \
  docker.io/grafana/grafana:latest

echo "stack 起來了：API :8000 / Prometheus :9090 / Grafana :3000"
echo "開 http://127.0.0.1:3000 看 dashboard（匿名免登入，注意是 127.0.0.1 不是 localhost）"
