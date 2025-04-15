# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container
COPY backend/src/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY scripts /scripts/

EXPOSE 8000

WORKDIR /app/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]