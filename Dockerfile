FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps:
# - poppler-utils: pdf2image needs pdftoppm
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
  && pip install -r /app/requirements.txt

# Optional: include MinerU CLI in the image (heavier).
# Enable with: --build-arg INSTALL_MINERU=1
ARG INSTALL_MINERU=0
RUN if [ "$INSTALL_MINERU" = "1" ]; then \
      # MinerU CLI 依賴 torch 來判斷/使用 device（即使你跑 CPU 亦需要 torch）
      # 注意：torch/torchvision 必須來自同一個 PyTorch wheel index，版本要對齊，否則可能出現
      # RuntimeError: operator torchvision::nms does not exist
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1+cpu" "torchvision==0.24.1+cpu" \
      # MinerU pipeline 需要 doclayout_yolo（否則會 ModuleNotFoundError）
      && pip install "doclayout-yolo" \
      # MinerU pipeline 需要 ultralytics（YOLOv8/YOLOv10 等模型依賴）
      && pip install "ultralytics" \
      && pip install "mineru==2.7.*" "opencv-python<4.12" "numpy==1.26.2"; \
    fi

COPY . /app

# Default runtime config (override via docker-compose env)
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    DEBUG=False \
    USE_VLLM_ASYNC=False \
    MINERU_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
