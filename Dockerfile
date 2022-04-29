# Base image
FROM anibali/pytorch:1.5.0-cuda10.2
# Root user permissions
USER root

# Update
RUN apt-get update && apt-get install -y build-essential

# Install mujoco
RUN sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN sudo apt install -y unzip
COPY mujoco/mjpro150_linux.zip mjpro150_linux.zip
RUN mkdir ~/.mujoco
RUN unzip mjpro150_linux.zip -d ~/.mujoco
COPY mujoco/mjkey.txt mjkey.txt
RUN sudo cp mjkey.txt ~/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mjpro150/bin
#RUN git clone https://github.com/openai/mujoco-py --branch 1.50.1.0
#RUN pip install -e ./mujoco-py
RUN pip install mujoco-py

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

# Install gym and dependencies
RUN apt-get update
RUN apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
RUN pip install gym[all]==0.17.2
RUN pip install pyvirtualdisplay > /dev/null 2>&1

# Install requirements
COPY requirements.txt /opt/app/requirements.txt
RUN pip install -r /opt/app/requirements.txt

# Copy code, uncomment this before build image
WORKDIR /home/user/workspace
#COPY . .