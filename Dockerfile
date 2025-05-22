# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the port FastAPI will use
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:test_app", "--host", "0.0.0.0", "--port", "8000"]
