# Используем официальный образ Ubuntu 18.04
FROM ubuntu:bionic

WORKDIR /home
ENV HOME=/home
VOLUME /data
EXPOSE 8888

# Устанавливаем необходимые пакеты
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl gnupg git vim python3-pip wget usbutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add our Debian package repository to your system:
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Обновляем список пакетов и устанавливаем libedgetpu1-std и python3-pycoral
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pycoral libedgetpu1-std python3-tflite-runtime && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# USB Acceleration setup guide
RUN git clone https://github.com/google-coral/pycoral && \
    cd pycoral && \
    bash examples/install_requirements.sh classify_image.py
