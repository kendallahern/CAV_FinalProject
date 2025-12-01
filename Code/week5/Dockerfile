# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install OS dependencies for Marabou + PyTorch
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files into the container
COPY . /app

# Upgrade pip and install Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip install torch torchvision numpy matplotlib scikit-learn z3-solver

# Optional: Install Marabou from GitHub if needed
# RUN git clone https://github.com/NeuralNetworkVerification/Marabou.git /app/Marabou
# RUN cd /app/Marabou && pip install -e .

# Default command
CMD ["bash"]
