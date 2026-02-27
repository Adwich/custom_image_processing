FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --upgrade pip && pip install .

# Preload default rembg models to reduce cold-start variance.
RUN python -c "from rembg import new_session; [new_session(m) for m in ('u2net_human_seg','isnet-general-use','u2net')]" || true

CMD ["python", "-m", "src.main"]
