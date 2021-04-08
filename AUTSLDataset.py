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
    Dataset for the AUTSL data
    '''
    
    def __init__(self, data_type, frame_interval = 1, data_path ="./dataset/" ):
        self.data_type = data_type
        self.frame_interval = frame_interval
        self.data_path = {"train": data_path + "train/"}        
        self.data_length = {"train":  len(glob(f"{self.data_path['train']}*_color.mp4"))}
        # self.data_length = {"train":  len(os.listdir(self.data_path["train"]))}
        self.label_path = {"train": data_path + "train_labels.csv"}
        self.labels = pd.read_csv(self.label_path[self.data_type],names = ["file_name", "label"])
        self.max = 0

#         self.max = self.get_max()
        
    def Describe(self):
        msg = "AUTSL Dataset\n"
        print(msg)
        
    def GetLabel(self, file_name):
        return self.labels[self.labels.file_name ==file_name]["label"].values[0]
    
    def GetVideoArray(self, file_name):
        cap = cv2.VideoCapture(self.data_path[self.data_type] + file_name)   # capturing the video from the given path
        video_arr = []
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % self.frame_interval == 0):
                # reshape to correct format
                # frame = frame.reshape(3,512,512)
                # normalising all frame to between 0 and 1
                # frame = frame/255
                # Converting to tensor
                frame =  torchvision.transforms.functional.to_tensor(frame).float()
                video_arr.append(frame)
                
        cap.release()

        
    
        if len(video_arr)<self.max:
            print("Video length is",len(video_arr),", Will add padding")
            # video_arr = F.pad(input = video_arr, pad = (0,1,1,1), mode = "constant", value = 0)
            empty_frame = torch.zeros((3,512,512))
            padding  = [empty_frame]*(self.max-len(video_arr))
            video_arr+=padding

        return torch.stack(video_arr)
        
    def __len__(self):
        return self.data_length[self.data_type]
    
    def __getitem__(self, index):
        '''
        Return 4D array consists of all the frame of the video image AND the label
        '''
        file_name = os.path.basename(glob(f"{self.data_path[self.data_type]}*_color.mp4")[index])
        # print(file_name)
        video_arr = self.GetVideoArray(file_name)
        label = self.GetLabel(file_name[:-10]) #slice to get just the name without file ext and file type
        return video_arr, label
        
    def get_max(self):
        max_length = 0
        for i in tqdm(range(len(self))):
            video, label = self[i]
            if video.shape[0]>max_length:
                max_length = video.shape[0]
        print(max_length)
        return max_length
    

        