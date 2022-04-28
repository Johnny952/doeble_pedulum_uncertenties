#!/bin/bash

docker run --name pendulum --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    -it -v $(pwd):/home/user/workspace \
    pendulum /bin/bash

#-v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -e DISPLAY=$DISPLAY \