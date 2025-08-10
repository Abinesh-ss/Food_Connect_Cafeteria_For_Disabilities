FROM python:3.10-slim

# Install system dependencies required for mediapipe runtime
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-imgproc-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose port for Render
EXPOSE 10000

# Start Flask with Gunicorn
CMD ["gunicorn", "flask.app:app", "-b", "0.0.0.0:10000"]
