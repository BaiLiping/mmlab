# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from os import path

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap

def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             img_metas,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)

def map_pointcloud_to_image(self,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False,
                            filter_lidarseg_labels: List = None,
                            lidarseg_preds_bin_path: str = None,
                            show_panoptic: bool = False) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """
    cam = self.nusc.get('sample_data', camera_token)
    pointsensor = self.nusc.get('sample_data', pointsensor_token)
    pcl_path = path.join(self.nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        if show_lidarseg or show_panoptic:
            gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
            assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'
            # Ensure that lidar pointcloud is from a keyframe.
            assert pointsensor['is_key_frame'], \
                'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
            assert not render_intensity, 'Error: Invalid options selected. You can only select either ' \
                                         'render_intensity or show_lidarseg, not both.'
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    # Second step: transform from ego to the global frame.
    poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))
    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    # Fourth step: transform from ego into the camera.
    cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities
    elif show_lidarseg or show_panoptic:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']
        gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
        semantic_table = getattr(self.nusc, gt_from)
        if lidarseg_preds_bin_path:
            sample_token = self.nusc.get('sample_data', pointsensor_token)['sample_token']
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
        else:
            if len(semantic_table) > 0:  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                    self.nusc.get(gt_from, pointsensor_token)['filename'])
            else:
                lidarseg_labels_filename = None
        if lidarseg_labels_filename:
            # Paint each label in the pointcloud with a RGBA value.
            if show_lidarseg:
                coloring = paint_points_label(lidarseg_labels_filename,
                                              filter_lidarseg_labels,
                                              self.nusc.lidarseg_name2idx_mapping,
                                              self.nusc.colormap)
            else:
                coloring = paint_panop_points_label(lidarseg_labels_filename,
                                                    filter_lidarseg_labels,
                                                    self.nusc.lidarseg_name2idx_mapping,
                                                    self.nusc.colormap)
        else:
            coloring = depths
            print(f'Warning: There are no lidarseg labels in {self.nusc.version}. Points will be colored according '
                  f'to distance from the ego vehicle instead.')
    else:
        # Retrieve the color from the depth.
        coloring = depths
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]
    return points, coloring, im
    
def render_pointcloud_in_image(self,
                               sample_token: str,
                               dot_size: int = 5,
                               pointsensor_channel: str = 'LIDAR_TOP',
                               camera_channel: str = 'CAM_FRONT',
                               out_path: str = None,
                               render_intensity: bool = False,
                               show_lidarseg: bool = False,
                               filter_lidarseg_labels: List = None,
                               ax: Axes = None,
                               show_lidarseg_legend: bool = False,
                               verbose: bool = True,
                               lidarseg_preds_bin_path: str = None,
                               show_panoptic: bool = False):
    """
    Scatter-plots a pointcloud on top of image.
    :param sample_token: Sample token.
    :param dot_size: Scatter plot dot size.
    :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :param out_path: Optional path to save the rendered figure to disk.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
    :param ax: Axes onto which to render.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param verbose: Whether to display the image in a window.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    if show_lidarseg:
        show_panoptic = False
    sample_record = self.nusc.get('sample', sample_token)
    # Here we just grab the front camera and the point sensor.
    pointsensor_token = sample_record['data'][pointsensor_channel]
    camera_token = sample_record['data'][camera_channel]
    points, coloring, im = self.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                        render_intensity=render_intensity,
                                                        show_lidarseg=show_lidarseg,
                                                        filter_lidarseg_labels=filter_lidarseg_labels,
                                                        lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                        show_panoptic=show_panoptic)
    # Init axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        if lidarseg_preds_bin_path:
            fig.canvas.set_window_title(sample_token + '(predictions)')
        else:
            fig.canvas.set_window_title(sample_token)
    else:  # Set title on if rendering as part of render_sample.
        ax.set_title(camera_channel)
    ax.imshow(im)
    ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
    ax.axis('off')
    # Produce a legend with the unique colors from the scatter.
    if pointsensor_channel == 'LIDAR_TOP' and (show_lidarseg or show_panoptic) and show_lidarseg_legend:
        # If user does not specify a filter, then set the filter to contain the classes present in the pointcloud
        # after it has been projected onto the image; this will allow displaying the legend only for classes which
        # are present in the image (instead of all the classes).
        if filter_lidarseg_labels is None:
            if show_lidarseg:
                # Since the labels are stored as class indices, we get the RGB colors from the
                # colormap in an array where the position of the RGB color corresponds to the index
                # of the class it represents.
                color_legend = colormap_to_colors(self.nusc.colormap, self.nusc.lidarseg_name2idx_mapping)
                filter_lidarseg_labels = get_labels_in_coloring(color_legend, coloring)
            else:
                # Only show legends for all stuff categories for panoptic.
                filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))
        if filter_lidarseg_labels and show_panoptic:
            # Only show legends for filtered stuff categories for panoptic.
            stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
            filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))
        create_lidarseg_legend(filter_lidarseg_labels, self.nusc.lidarseg_idx2name_mapping, self.nusc.colormap,
                               loc='upper left', ncol=1, bbox_to_anchor=(1.05, 1.0))
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()

def _plot_points_and_bboxes(self,
                            pointsensor_token: str,
                            camera_token: str,
                            filter_lidarseg_labels: Iterable[int] = None,
                            lidarseg_preds_bin_path: str = None,
                            with_anns: bool = False,
                            imsize: Tuple[int, int] = (640, 360),
                            dpi: int = 100,
                            line_width: int = 5,
                            show_panoptic: bool = False) -> Tuple[np.ndarray, bool]:
    """
    Projects a pointcloud into a camera image along with the lidarseg labels. There is an option to plot the
    bounding boxes as well.
    :param pointsensor_token: Token of lidar sensor to render points from and lidarseg labels.
    :param camera_token: Token of camera to render image from.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
                                   or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param with_anns: Whether to draw box annotations.
    :param imsize: Size of image to render. The larger the slower this will run.
    :param dpi: Resolution of the output figure.
    :param line_width: Line width of bounding boxes.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels.
    :return: An image with the projected pointcloud, lidarseg labels and (if applicable) the bounding boxes. Also,
             whether there are any lidarseg points (after the filter has been applied) in the image.
    """
    points, coloring, im = self.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                        render_intensity=False,
                                                        show_lidarseg=not show_panoptic,
                                                        show_panoptic=show_panoptic,
                                                        filter_lidarseg_labels=filter_lidarseg_labels,
                                                        lidarseg_preds_bin_path=lidarseg_preds_bin_path)
    # Prevent rendering images which have no lidarseg labels in them (e.g. the classes in the filter chosen by
    # the users do not appear within the image). To check if there are no lidarseg labels belonging to the desired
    # classes in an image, we check if any column in the coloring is all zeros (the alpha column will be all
    # zeroes if so).
    if (~coloring.any(axis=0)).any():
        no_points_in_im = True
    else:
        no_points_in_im = False
    if with_anns:
        # Get annotations and params from DB.
        impath, boxes, camera_intrinsic = self.nusc.get_sample_data(camera_token, box_vis_level=BoxVisibility.ANY)
        # We need to get the image's original height and width as the boxes returned by get_sample_data
        # are scaled wrt to that.
        h, w, c = cv2.imread(impath).shape
        # Place the projected pointcloud and lidarseg labels onto the image.
        mat = plt_to_cv2(points, coloring, im, (w, h), dpi=dpi)
        # Plot each box onto the image.
        for box in boxes:
            # If a filter is set, and the class of the box is not among the classes that the user wants to see,
            # then we skip plotting the box.
            if filter_lidarseg_labels is not None and \
                    self.nusc.lidarseg_name2idx_mapping[box.name] not in filter_lidarseg_labels:
                continue
            c = self.get_color(box.name)
            box.render_cv2(mat, view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=line_width)
        # Only after points and boxes have been placed in the image, then we resize (this is to prevent
        # weird scaling issues where the dots and boxes are not of the same scale).
        mat = cv2.resize(mat, imsize)
    else:
        mat = plt_to_cv2(points, coloring, im, imsize, dpi=dpi)
    return mat, no_points_in_im