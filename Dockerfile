# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM python:3.11-bullseye AS builder

WORKDIR /app

# Install dependencies
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# ============================================
# Stage 2: Runtime - Final image
# ============================================
FROM python:3.11-slim-bullseye AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy downloaded models (if stored in specific location)
# COPY --from=builder /root/.cache /root/.cache

# Port config
ENV FAST_API_PORT=9090
EXPOSE 9090

# Copy source code (changes frequently)
COPY *.py .
COPY ./ui ./ui
# Install models (if any)
COPY install_deps.py .
RUN python3 install_deps.py

# Start the FastAPI server
CMD python3 server.py --port ${FAST_API_PORT}
