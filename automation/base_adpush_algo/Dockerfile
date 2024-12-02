# Base image
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY app.py .
COPY train_250k_query_emb.csv .

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
