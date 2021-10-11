from Project_5.inference import inference
from nuscenes_interface import NuScenes
import os
import copy
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread
import numpy as np
import torch

# Video Generating function
def imagetovideo(image_path, num_images, video_path):
    image_folder = image_path # make sure to use your folder
    images = [img for img in os.listdir(image_folder)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    # set the flip per second varible
    fps = 5
    # setting the frame width, height width
    height, width, layers = frame.shape  
    # save video as mp4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height)) 
  
    # Appending the images to the video one by one
    for i in range(num_images):
        video.write(cv2.imread(os.path.join(image_folder, '{}.png'.format(i)))) 
      
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated

# generate video from the readout data
def generate_video(nuscenes_data, root_directory_for_out_path):
    # readout the scenes from input data
    scenes=nuscenes_data.scene
    for scene in scenes:
        # read out the scene name in order to get to the directory
        scene_name=scene['name']
        # generate the input path for this scene
        image_path = os.path.join(root_directory_for_out_path,scene_name)
        num_of_images = len(os.listdir(image_path))
        video_path = image_path+'/{}.mp4'.format(scene_name)
        imagetovideo(image_path, num_of_images,video_path)

def read_out_time_stamp_information(inference_result):
    inference_result['timestamp']=[]
    for file_name in inference_result['file_name']:
        timestamp=file_name[-16:]
        inference_result['timestamp'].append(timestamp)

# extract data from the database
def generate_visualization(nuscenes_data,inference_result, root_directory_for_out_path):
    '''
    nuscenes_data is the database readout, it is a NuScenes object as defined by data_extraction file
    root_directory_for_out_path is the input parameter from configure file
    '''
    # read out scenes of this dataset
    scenes=nuscenes_data.scene
    # read out frames of this dataset
    frames=nuscenes_data.sample
    # read out time stamp information from inference data
    read_out_time_stamp_information(inference_result)
    
    for scene in scenes:
        # get the token of this scene
        scene_token=scene['token']
        # get the name of this scene
        scene_name=scene['name']
        # generate the out file directory for this scene
        out_file_directory_for_this_scene = os.path.join(root_directory_for_out_path,scene_name)
        # if the directory exist, then delete and make a new directory
        if os.path.exists(out_file_directory_for_this_scene):
            print('erasing existing data')
            shutil.rmtree(out_file_directory_for_this_scene, ignore_errors=True)
        os.mkdir(out_file_directory_for_this_scene)
        
        # get all the frames associated with this scene
        frames_for_this_scene = []
        for frame in frames:
            if frame['scene_token']==scene_token:
                frames_for_this_scene.append(frame)
        
        # set the frames in corret order
        # notice that NuScenes database does not provide the numerical ordering of the frames
        # it provides previous and next frame token information
        unordered_frames = copy.deepcopy(frames_for_this_scene)
        ordered_frames=[]
        # looping until the unordered frame is an empty set
        while len(unordered_frames)!=0:
            for current_frame in unordered_frames:
                # if it is the first frame
                if current_frame['prev']=='':
                    ordered_frames.append(current_frame)
                    # set current token
                    current_frame_token_of_current_scene = current_frame['token']
                    unordered_frames.remove(current_frame)
        
                # find the next frame of current frame
                if current_frame['prev']==current_frame_token_of_current_scene:
                    ordered_frames.append(current_frame)
                    # reset current frame
                    current_frame_token_of_current_scene=current_frame['token']
                    unordered_frames.remove(current_frame)

        inference_result_search_list = copy.deepcopy(inference_result)
        # get the data from ordered frame list
        count=0
        for frame in ordered_frames:
            frame_timestap = frame['timestamp']
            for i in range(len(inference_result_search_list['timestamp'])):
                if inference_result_search_list['timestamp'][i] == frame_timestap:
                    inference_result_boxes_3d =  inference_result_search_list['boxes_3d']
                    inference_result_labels_3d = inference_result_search_list['labels_3d']
    
            # notice this is a customized function, so please do not use code from the official dev-kit   
            nuscenes_data.render_inference_sample(inference_result_boxes_3d, inference_result_labels_3d,frame['token'],out_path=out_file_directory_for_this_scene+'/{}.png'.format(count),verbose=False)        
            count+=1
            plt.close('all')