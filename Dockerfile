# Important - 3.13 does not support Gensim (no wheels?)
FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set up work directory
WORKDIR /app

# Copy code
COPY . .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint
CMD ["python", "main.py"]