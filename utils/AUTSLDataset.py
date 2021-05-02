import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from glob import glob
from tqdm import tqdm

import torchvision
from torchvision import models
from torchsummary import summary


import os
import pandas as pd
import math



class AUTSLDataset(Dataset):
    '''
    Pytorch Dataset for the AUTSL data

    Parameters
    ----------
    data_type : str : "train" or "val" datatype
    max_frame_no : int : maximum number of frame to pad, -1 if padding is not needed
    num_class : int : maximum number of class
    frame_interval : int : will reduce the frame rate by this frame interval
    data_path : str : path to the data

    Returns
    -------
    Dataset
        Dataset for the AUTSL data
    '''
    
    def __init__(self, data_type, max_frame_no, num_class=226 ,frame_interval = 1, data_path ="./dataset/" ):
        self.data_type = data_type
        self.frame_interval = frame_interval
        self.data_path = data_path+self.data_type+"/"
        self.label_path = data_path + self.data_type+"_labels.csv"

        df = pd.read_csv(self.label_path,names = ["file_name", "label"])
        self.num_class = num_class  
        self.df = df[df.label < self.num_class]
        self.df["file"] = self.df.file_name + "_color.mp4"
        
        self.file_ls = self.df.file.to_list()
        self.data_length = len(self.file_ls)
    
        self.max_frame_no = max_frame_no
        self.device = "cuda"

        
    def Describe(self):
        msg = "AUTSL Dataset\n"
        print(msg)
    
        
    def GetLabel(self, file):
        return self.df[self.df.file ==file]["label"].values[0]
    
    def GetVideoArray(self, file_name):

        cap = cv2.VideoCapture(self.data_path + file_name)   # capturing the video from the given path
        video_arr = []
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % self.frame_interval == 0):
                # Converting to tensor
                frame =  torchvision.transforms.functional.to_tensor(frame).float().to(self.device)
                frame = frame.unsqueeze(0)
                frame =  F.interpolate(frame, (256,256), mode='bilinear')
                frame = frame.squeeze(0)
                video_arr.append(frame)
        cap.release()

    
        if len(video_arr)<self.max_frame_no:
            empty_frame = torch.zeros((3,256,256)).to(self.device)
            padding  = [empty_frame]*(self.max_frame_no-len(video_arr))
            video_arr+=padding

        return torch.stack(video_arr) 
        
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index):
        '''
        Return 4D array consists of all the frame of the video image AND the label
        '''
        file_name = os.path.basename(self.file_ls[index])
        # print(file_name)
        video_arr = self.GetVideoArray(file_name)
        
        label = self.GetLabel(file_name)
        return video_arr, label
        
    def get_max(self):
        max_length = 0
        for i in tqdm(range(len(self))):
            video, label = self[i]
            if video.shape[0]>max_length:
                max_length = video.shape[0]
        return max_length
    
