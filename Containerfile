# 推論服務容器（GPU）。模型不烤進 image——用 -v 在 run 時 mount（與映像解耦）。
# torch cu128 wheel 自帶 CUDA runtime 函式庫；GPU「裝置 + 驅動」由 CDI passthrough 提供
# （podman run --device nvidia.com/gpu=all），所以基底用 python-slim 即可，不需 nvidia/cuda。
FROM python:3.12-slim

WORKDIR /app

# 只裝「推論」需要的（不含 jupyter/matplotlib/datasets 等開發用套件 → image 小一點）。
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" numpy prometheus-client

# 只複製服務需要的程式碼（模型走 mount）
COPY src/ ./src/
COPY serve/ ./serve/

ENV ARTIFACTS=/app/artifacts
EXPOSE 8000

# 0.0.0.0 才能從容器外連入
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
