# ---- Base image ----
FROM python:3.11-slim

# ---- System dependencies ----
# Required for:
# - pandas / numpy wheels
# - dvc[s3]

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    build-essential \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*


# ---- Set working directory ----
WORKDIR /app

# ---- Python dependencies ----
# Copy requirements first to leverage Docker layer caching
COPY docker/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Copy project ----
COPY . .

# ---- Environment variables (safe defaults) ----
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_EXPERIMENT_NAME=HostScope

# ---- Pull DVC-tracked data at container start ----
ENTRYPOINT ["bash", "-c", "dvc pull"]

# ---- Default command (analysis container) ----
CMD ["python"]