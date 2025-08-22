# Use lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Use gunicorn to serve Flask
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:app"]
