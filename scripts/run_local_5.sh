#!/bin/bash

docker run --name pendulum --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=5 -it -v $(pwd):/home/user/workspace pendulum /bin/bash