FROM python:3.11-slim

WORKDIR /app

# Install system dependencies minimal
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python deps tanpa cache pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode
COPY . .

# Buat folder cache_model sebagai cache Huggingface model & tokenizer
RUN mkdir -p /app/cache_model

# Download dan simpan model + tokenizer di /app/cache_model saat build
RUN python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; \
    T5Tokenizer.from_pretrained('t5-small', cache_dir='/app/cache_model'); \
    T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir='/app/cache_model')"

# Set environment variable agar transformers pakai folder cache_model kita
ENV TRANSFORMERS_CACHE=/app/cache_model
ENV HF_HOME=/app/cache_model

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
