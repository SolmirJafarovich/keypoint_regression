FROM ubuntu:20.04

WORKDIR /home
ENV HOME=/home
VOLUME /data
EXPOSE 8888

# System environment
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/home/.venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl gnupg git vim wget usbutils build-essential libffi-dev \
    python3 python3-dev python3-venv python3-pip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add Coral repo
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install Coral Python libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pycoral libedgetpu1-std python3-tflite-runtime \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with uv (using lockfile)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/home/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/home/pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Устанавливаем OpenCV и его зависимости
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-opencv libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy source and install the project itself
ADD . /home
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Clear entrypoint; container will expect a command
ENTRYPOINT []
