# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Internal)
EXPOSE 8080

# Run uvicorn on port 8080
CMD ["uvicorn", "task3:app", "--host", "0.0.0.0", "--port", "8080"]
