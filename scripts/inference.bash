docker build -t keypoint_regression .
docker run --rm -it \
    --privileged \
    --device /dev/video0:/dev/video0 \
    --device /dev/bus/usb:/dev/bus/usb \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    keypoint_regression \
    bash 
