FROM python:3.10-slim

# Install system dependencies for mediapipe and dlib runtime (not compilation)
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install precompiled dlib wheel first (fast)
RUN pip install --no-cache-dir dlib==19.24.2 --only-binary=:all:

# Then install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000
CMD ["gunicorn", "flask.app:app", "-b", "0.0.0.0:10000"]
