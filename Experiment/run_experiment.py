from nuscenes_interface import NuScenes
from utils import generate_video,generate_visualization, reformat_inference_result
import os
import shutil
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
import argparse
import numpy as np
from copy import deepcopy
from inference import generate_inference_data
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_directory_for_dataset', help='root directory for the entire dataset, including mini, trainval and test')
    parser.add_argument('--root_directory_for_visualization_out_path',help='output result file for visualization')
    parser.add_argument('--inference_result', help='output result file for inference in pickle format')
    parser.add_argument('--dataset_version', help='v1.0-mini, v1.0-trainval or v1.0-test')
    parser.add_argument('--inference_network_pretrained_model', help='pth file')   
    parser.add_argument('--inference_network_config', help='test config file path')
    parser.add_argument('--test_data_point_cloud_data_file', help='wehre to find the test point cloud data')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_set_directory = os.path.join(args.root_directory_for_dataset,args.data_set_version)
    # clear the out folder if it exist
    if os.path.exists(args.root_directory_for_visualization_out_path):
        user_input = input("visualization result already exist, enter 'yes' if you want to erase this folder and regenerate visualization result \n  enter 'no' if you want to take a look at the folder first: \n")
        if user_input == 'yes':
            # delete the folder
            shutil.rmtree(args.root_directory_for_visualization_out_path, ignore_errors=True)
            os.mkdir(args.root_directory_for_visualization_out_path)
        elif user_input == 'no':
            pass
        else:
            raise ValueError('invalid input')
    else:
        # make directory for visualization the out file
        os.mkdir(args.root_directory_for_visualization_out_path)
        # verbose set to true if want to show image, otherwise set to false, image would be save to out directory
        nuscenes_data = NuScenes(version = 'v1.0-test', dataroot=args.data_set_directory, verbose=False)
        # extract data from the database
        # database can be either mini, train/validation or test
        if os.path.exists(args.inference_result):
            print('inference result already exist, use existing inference output')
        else:
            print('no existing inferecen result, generate inference data')
            generate_inference_data(args.inference_network_config, args.inference_network_pretrained_model, args.test_data_point_cloud_data_file,args.inference_result)

        # read in the inference result
        with open(args.inference_result, 'rb') as f:
            inference_result = pickle.load(f)
        generate_visualization(nuscenes_data, inference_result, args.root_directory_for_visualization_out_path)
        # generate video from the extracted data
        generate_video(nuscenes_data,args.root_directory_for_visualization_out_path)


if __name__ == '__main__':
    main()
