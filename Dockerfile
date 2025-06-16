# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install system packages 
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Start both FastAPI and Gradio using uvicorn and Python
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & python utils/ui.py"]
