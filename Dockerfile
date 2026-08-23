# syntax=docker/dockerfile:1

# ---------- base: ไลบรารีภาพ + numpy (เบา ใช้รันเทสต์ได้เลย ไม่ต้องมี torch) ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt
COPY pytest.ini ./
COPY ekg_rpeak ./ekg_rpeak
COPY webapp ./webapp
COPY tests ./tests

# ---------- test: ชุดทดสอบเร็ว (ใช้โมเดลจำลอง ไม่ต้องโหลด torch/weights) ----------
FROM base AS test
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
CMD ["pytest", "-v"]

# ---------- runtime: เพิ่ม torch (CPU) + ultralytics สำหรับรันงานจริง ----------
FROM base AS runtime
COPY requirements-runtime.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install --no-cache-dir -r requirements-runtime.txt -r requirements-dev.txt
ENV YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/mpl
ENTRYPOINT ["python", "-m", "ekg_rpeak.cli"]
CMD ["--help"]

# ---------- web: หน้าเว็บดูผล ----------
FROM runtime AS web
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
EXPOSE 8000
ENTRYPOINT []
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000"]
