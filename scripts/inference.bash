docker build -t keypoint_regression .
docker run --rm -it \
    --privileged \
    --device /dev/video0:/dev/video0 \
    --device /dev/bus/usb:/dev/bus/usb \
    -v "$(pwd)":/home \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    keypoint_regression \
    uv run -m scripts.evaluate --reg data/checkpoints/regressor --cl data/checkpoints/classifier
