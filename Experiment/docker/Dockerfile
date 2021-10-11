FROM scrin/dev-spconv:latest

# for users insinde mainland China, change the server to Tsinghua
RUN sudo echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list
RUN sudo apt update
RUN apt upgrade -y
RUN apt install -y python3-pip
RUN python3 -m pip install --upgrade pip
RUN apt install -y libgl1-mesa-dev
RUN apt install -y git
# change the pip repository to Tsinghua
#RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# install git
RUN apt install -y git
RUN apt install -y openssh-client

# Install MMDetection3D

RUN mkdir /mmdetection3D
ENV FORCE_CUDA="1"
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d
RUN pip install nuscenes-devkit
WORKDIR /mmdetection3d
# for machines without cuda, please compling within the container
RUN pip install -v -e .
RUN pip install numpy==1.21.2
