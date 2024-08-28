# Use a more complete Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
#WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    zlib1g-dev \
    libjpeg62-turbo-dev \
    liblcms2-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libatlas-base-dev \
    python3-dev \
    libtiff-dev \
    libfreetype6-dev \
    libpng-dev \
    libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]