# StoryRec — two-stage personalized recommendation engine
FROM python:3.11-slim

WORKDIR /app

# libgomp is required by LightGBM; build tools keep faiss/torch wheels happy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the full pipeline at build time so the image ships ready to serve.
# Comment this out to train at runtime instead.
RUN python pipeline.py

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; \
    urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
