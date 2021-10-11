import pickle

from inference import inference

inference_result='/home/zhubinglab/Desktop/NuScenes_Project/inference_result.pkl'
with open(inference_result, 'rb') as f:
    inference_data = pickle.load(f)


for i in range(len(inference_data['file_name'])):
    #print('FILE NAME:')
    print(inference_data['file_name'][i])
    #print('BBOXES')
    #print(inference_data['bboxes'][i])
    #print('LABELS')
    #print(inference_data['labels'][i])
print(len(inference_data['file_name']))
