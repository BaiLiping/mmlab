# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path
import glob
from tqdm import tqdm
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
import pickle
import argparse
import numpy as np
import torch
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_result', help='output result file for inference in pickle format')
    parser.add_argument('--inference_network_pretrained_model', help='pth file')   
    parser.add_argument('--inference_network_config', help='test config file path')
    parser.add_argument('--root_directory_for_test_data_point_cloud_data_file', help='wehre to find the test point cloud data')

    args = parser.parse_args()
    return args

def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def init_model(config, checkpoint, device='cuda:0'):
    """Initialize a model from config file, which could be a 3D detector or a
    3D segmentor.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Device to use.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    convert_SyncBN(config.model)
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = config.class_names
        if 'PALETTE' in checkpoint['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        bbox and label
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        # t his is a work around, just set ann_info to be zero
        # TODO this annotation data need to be something, generate it from the evaluation data.
        ann_info=[],
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    bboxes_3d = result[0]['pts_bbox']['boxes_3d']
    labels_3d = result[0]['pts_bbox']['labels_3d']
    return bboxes_3d, labels_3d

def removeprefix(input_string, prefix):
    return input_string[len(prefix):]

def generate_inference_data(config_file, pretrained_network, root_directory_for_pcd_path,inference_out_path):
    model = init_model(config_file, pretrained_network)
    lidar_inference_result = {}
    lidar_inference_result['file_name']=[]
    lidar_inference_result['bboxes']=[]
    lidar_inference_result['labels']=[]
    direcotry_for_sample_data = path.join(root_directory_for_pcd_path, 'samples/LIDAR_TOP')
    directory_for_sweep_data = path.join(root_directory_for_pcd_path, 'sweeps/LIDAR_TOP')
    #notice the order would be reversed after the glob.glob function
    sample_pcd_list=glob.glob(path.join(direcotry_for_sample_data,'*.pcd.bin'))
    sweeps_pcd_list=glob.glob(path.join(directory_for_sweep_data,'*.pcd.bin' ))
    pcd_list=sample_pcd_list+sweeps_pcd_list
    for i in tqdm(range(len(pcd_list))):
    #for pcd in pcd_list:
        pcd=pcd_list[i]
        bboxes_3d, labels_3d=inference(model,pcd)
        pcd=removeprefix(pcd, root_directory_for_pcd_path+'/')
        lidar_inference_result['file_name'].append(pcd)
        lidar_inference_result['bboxes'].append(bboxes_3d)
        lidar_inference_result['labels'].append(labels_3d)
    inference_out = open(inference_out_path,'wb')
    pickle.dump(lidar_inference_result, inference_out)

if __name__ == '__main__':
    args = parse_args()
    generate_inference_data(args.inference_network_config,args.inference_network_pretrained_model, args.root_directory_for_test_data_point_cloud_data_file,args.inference_result)

