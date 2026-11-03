FROM python:3.11-slim

# OpenCV (pulled in by mediapipe) needs libGL/glib at import time
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Models pinned by sha256 (the exact files the bake-off was tuned with, 2026-11-02)
RUN mkdir -p models \
    && curl -sSfL -o models/selfie_multiclass.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite \
    && curl -sSfL -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
    && echo "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0  models/selfie_multiclass.tflite" | sha256sum -c - \
    && echo "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  models/face_landmarker.task" | sha256sum -c -

COPY cutout.py framing.py server.py ./
ENV PYTHONUNBUFFERED=1 PORT=8080
EXPOSE 8080
CMD ["python", "server.py"]
