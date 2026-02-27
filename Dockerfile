# syntax = docker/dockerfile:1

## Uncomment the version of python you want to test against
# FROM python:3.10-slim
# FROM python:3.11-slim
# FROM python:3.12-slim
# FROM python:3.13-slim
# FROM python:3.14-slim
FROM python:3.14-bookworm


# Set the working directory to /app
WORKDIR /app/

RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    sox \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Copy and install the requirements
COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
RUN touch /app/spych/__init__.py
RUN touch README.md

RUN pip install -r requirements.txt

# Drop into a shell by default
CMD ["/bin/bash"]
