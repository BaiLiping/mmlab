1. [Schema of NuScenes Dataset](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)


2. system setup. Lidar 3D inference require c++ compliation. After a lot of trial and error, here is the system that works for us.
   A not for nvidia driver installation. read the official document for uninstall, as if you purge things manually, there would be broken
   linkes and packages left. Use the cuda installation guide for the driver installation. 
- ![current system](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/current_system1.png)
- ![current system](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/current_system2.png)
- ![current system](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/current_system3.png)
- Alternatively, here is another setup that can work
- ![alternative setup](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/alternative_system_setup.png)


3. visualization for the annotation (ground truth)
- ![1](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/scene1.gif)
- ![1](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/scene2.gif)
- ![1](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/scene3.gif)
- ![1](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/scene4.gif)
- ![1](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/scene5.gif)
4. Alternatively one can install a docker image with the Dockerfile provided. Or you can pull an image from `docker push bailiping/mmlab:v2.0`
- ![1](https://raw.githubusercontent.com/BaiLiping/project_pictures/master/dockerimage.png)
5. enable Nvidia GPU driver interface, please follow the intruction provided [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
```
Set `nvidia-container-runtime` as default docker runtime by edit the `/etc/docker/daemon.json` file:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```
restart docker `sudo systemctl restart docker`

6. to run the docker image

```
sudo docker run  --gpus all -it -v '/home/zhubinglab/Desktop/NuScenes_Project':/root/NuScenes_Project  --name mmlab bailiping/mmlab:v2.0 bash
```

7. inside the docker, got to mmdetection3d directory
```
mkdir checkpoint
cd checkpoint
wget https://download.openmmlab.com/mmdetection3d/v0.1.0_models/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth
```

we use this centerpoint model for our inference, yet you can download any other pretrained models. For detailed discription, please read [mmlab documentation](https://mmdetection3d.readthedocs.io/en/latest/model_zoo.html)


6. CenterPoint related docker file can be found at [this](https://github.com/BaiLiping/CenterPointDocker) repository

7. After a lot of trial and error, we settled on the following specifications for our experiments:


8. for installing mmlab3d, please refer to their [website]()
