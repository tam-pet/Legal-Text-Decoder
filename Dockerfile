# Use Python 3.10 with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/log

# Set Python path
ENV PYTHONPATH=/app/src

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "========================================"\n\
echo "LEGAL TEXT DECODER - Deep Learning Project"\n\
echo "========================================"\n\
echo ""\n\
\n\
# Step 1: Data Preprocessing\n\
echo "[1/4] Running data preprocessing..."\n\
python src/a01_data_preprocessing.py\n\
\n\
# Step 2: Training\n\
echo "[2/4] Training models..."\n\
python src/a02_training.py\n\
\n\
# Step 3: Evaluation\n\
echo "[3/4] Evaluating models..."\n\
python src/a03_evaluation.py\n\
\n\
# Step 4: Inference Demo\n\
echo "[4/4] Running inference demo..."\n\
python src/a04_inference.py --text "A Szolgáltató fenntartja a jogot, hogy a szolgáltatást bármikor módosítsa vagy megszüntesse."\n\
\n\
echo ""\n\
echo "========================================"\n\
echo "Pipeline completed successfully!"\n\
echo "========================================"\n\
' > /app/run_pipeline.sh && chmod +x /app/run_pipeline.sh

# Default command
CMD ["/bin/bash", "/app/run_pipeline.sh"]
