python3 inference.py --root_directory_for_dataset='/home/zhubing/Desktop/NuScenes_Project/data'\
                     --root_directory_for_visualization_out_path='/home/zhubing/Desktop/NuScenes_Project/visualization_result'\
                     --inference_result='/home/zhubinglab/Desktop/NuScenes_Project/inference_result.pkl'\
                     --dataset_version='v1.0-test'
                     --inference_network_pretrained_model='/home/zhubinglab/Desktop/Radar_Perception_Project/Project_5/pretrained_networks/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth'\
                     --inference_network_config='/home/zhubinglab/Desktop/Radar_Perception_Project/Project_5/configs/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py' \
                     --test_data_point_cloud_data_file='/home/zhubinglab/Desktop/NuScenes_Project/data/sweeps/LIDAR_TOP'
    