# Use an official lightweight Python image
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/src

# Install system dependencies if needed, copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

EXPOSE 8000

# Run Uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
