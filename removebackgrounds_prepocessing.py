import torch
from utils.segmentation import segment_img_rm_bg
from torch.nn import functional as F
import cv2
import torchvision
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
device = torch.device('cpu')
# device = torch.device('cuda')

deeplabmodel = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).to(device).eval()

dataset_path = './dataset/'

def segment_and_replace_video(filepath, out_dir):
    '''
    Preprocess to remove video background
    1. reads the video
    2. reduces video size to 256x256 and fps to 15
    3. removes background
    4. writes to outdir
    5. deletes original video

    Parameters
    ----------
    filepath : str : path to file
    out_dir : str : path to the output

    Returns
    -------
    None
    '''
    filename = os.path.basename(filepath)
    cap = cv2.VideoCapture(filepath)   # capturing the video from the given path    
    video_arr = []
    out = None
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        
        frame_interval = 2
        if not out:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')            
            #0x7634706d
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"out: {f'{out_dir}/{filename[:-4]}.mp4'}")
            out = cv2.VideoWriter(f'{out_dir}/{filename[:-4]}.mp4', fourcc, int(fps/frame_interval), (256, 256))

        if (ret != True):
            break

        #half fps by frame_interval
        if (frameId % frame_interval == 0):
            # Converting to tensor
            frame =  torchvision.transforms.functional.to_tensor(frame).float().to(device)
            frame = frame.unsqueeze(0)
            #reduces img size to 256x256
            frame =  F.interpolate(frame, (256,256), mode='bilinear')
            frame = frame.squeeze(0)
            frame_nobg = segment_img_rm_bg(frame, deeplabmodel, device)
            #write to output
            out.write((frame_nobg*255).astype(np.uint8))
    
    #release memory
    cap.release()    
    if out:
        out.release()  
    
    #delete original video
    if os.path.isfile(filepath):
        os.remove(filepath) 


folders = ['train', 'val']
#look through dataset/train and dataset/val, output to dataset/nobg/train and dataset/nobg/val the videos with backgrounds removed
for folder in folders:
    names = glob(f'{dataset_path}{folder}/*[!_nobg!_depth].mp4')
    for name in names:
        segment_and_replace_video(name, f'{dataset_path}nobg/{folder}')
        print(name)