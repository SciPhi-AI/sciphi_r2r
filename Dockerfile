FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ musl-dev curl libffi-dev gfortran libopenblas-dev \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN pip install --no-cache-dir poetry


# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /app/py
WORKDIR /app/py
COPY py/pyproject.toml /app/py/pyproject.toml

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --extras "core ingestion-bundle" --no-root \
    && pip install --no-cache-dir gunicorn uvicorn

# Create the final image
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add poppler to PATH
ENV PATH="/usr/bin:${PATH}"

# Debugging steps
RUN echo "PATH: $PATH"
RUN which pdfinfo
RUN pdfinfo -v

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Expose the port and set environment variables
ARG FUSE_PORT=8000
ARG FUSE_HOST=0.0.0.0
ENV FUSE_PORT=$FUSE_PORT
ENV FUSE_HOST=$FUSE_HOST
EXPOSE $FUSE_PORT

COPY py /app
# Copy the application and config
COPY py/core /app/core
COPY py/fuse /app/fuse
COPY py/shared /app/shared
COPY py/fuse/fuse.toml /app/fuse.toml
COPY py/pyproject.toml /app/pyproject.toml

# Run the application
CMD ["sh", "-c", "uvicorn core.main.app_entry:app --host $FUSE_HOST --port $FUSE_PORT"]
