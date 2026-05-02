# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file and install the Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Django project into the container
COPY . .

# Build the database tables
RUN python manage.py migrate

# Expose port 7860 (This is the mandatory port for Hugging Face Spaces)
EXPOSE 7860

# Start the Django server using Gunicorn on port 7860
CMD ["gunicorn", "traffic_monitor.wsgi:application", "--bind", "0.0.0.0:7860"]