# syntax = docker/dockerfile:1

## Uncomment the version of python you want to test against
# FROM python:3.11-bookworm
FROM python:3.12-bookworm
# FROM python:3.13-bookworm

## Can not test fully or build docs on 3.15 since kokoro does not support it yet
# FROM python:3.14-bookworm


# Set the working directory to /app
WORKDIR /app/

# Copy and install the requirements
COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
RUN mkdir -p /app/spych
RUN touch /app/spych/__init__.py

RUN pip install -r requirements.txt
# Install chatterbox-tts for testing and documentation purposes
RUN pip install chatterbox-tts

# Drop into a shell by default
CMD ["/bin/bash"]
