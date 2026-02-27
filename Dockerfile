# syntax = docker/dockerfile:1

## Uncomment the version of python you want to test against
FROM python:3.11-bookworm
FROM python:3.12-bookworm
FROM python:3.13-bookworm
FROM python:3.14-bookworm


# Set the working directory to /app
WORKDIR /app/

# Copy and install the requirements
COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
RUN mkdir -p /app/spych
RUN touch /app/spych/__init__.py

RUN pip install -r requirements.txt

# Drop into a shell by default
CMD ["/bin/bash"]
