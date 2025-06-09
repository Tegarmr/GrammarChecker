FROM python:3.11-slim

# Buat direktori kerja
WORKDIR /grammarchecker

# Install Git karena transformers sering ambil repo dari GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Copy file requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh isi project ke image
COPY . .

# Buat folder cache model
RUN mkdir -p /grammarchecker/app/cache_model

# Download model dan tokenizer ke folder cache
RUN python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; \
    T5Tokenizer.from_pretrained('t5-small', cache_dir='app/cache_model'); \
    T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir='app/cache_model')"

# Set environment variable agar transformers cache-nya di sana
ENV TRANSFORMERS_CACHE=/grammarchecker/app/cache_model
ENV HF_HOME=/grammarchecker/app/cache_model

EXPOSE 8000

# Jalankan app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
