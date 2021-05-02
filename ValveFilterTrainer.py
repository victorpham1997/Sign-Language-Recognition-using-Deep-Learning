import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.nn import functional as F
from torchsummary import summary
from glob import glob

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

import argparse
import os
from ModelClasses import *
import cv2
import pandas as pd
from utils.AUTSLDataset import AUTSLDatasetROI
from utils.train import *

import warnings 
warnings.filterwarnings("ignore")


### ARGPARSE SECTION
def checkPosInteger(value, valueType):
	try:
		ivalue = int(value)
		if ivalue <= 0:
			raise argparse.ArgumentTypeError("{} : {} is not a valid input!".format(valueType, value))
		return ivalue
	except:
		raise argparse.ArgumentTypeError("{} : {} is not a valid input!".format(valueType, value))
		
def dir_path(string):
	if os.path.isdir(string):
		return string
	else:
		os.mkdir(string)
		print ("{} directory made!".format(string))

parser = argparse.ArgumentParser(description='Valve Filter Model Trainer')
parser.add_argument('--n_epochs', type=lambda x: checkPosInteger(x, "n_epochs"), default=5, help="Number of epochs")
parser.add_argument('--lr', type=float, default=1e-5, help="Learning Rate")
parser.add_argument('--model_path', type=dir_path, default="./model", help="Input Model path for saving purposes")
parser.add_argument('--d', type=bool, default=True, help="Use Cuda or not")


args = parser.parse_args()

# Define Parameter here:
n_epochs = args.n_epochs
lr = args.lr
model_path = args.model_path
device = "cuda" if args.d else "cpu"
model_type = "baseline"
steps = 0 
print_every = 50
validate_every = 50


### END OF ARGPARSE
    
torch.manual_seed(0)

batch_size = 2

trainROI_autsl = AUTSLDatasetROI("train", int(156/2), frame_interval=4, file_percentage=0.3)
valROI_autsl = AUTSLDatasetROI("val", int(114/2), frame_interval=4, file_percentage=0.3)
trainroiloader = DataLoader(trainROI_autsl, batch_size=batch_size, shuffle=True)
valroiloader = DataLoader(valROI_autsl, batch_size=batch_size, shuffle=True)

# Define model instance 
model = BaselineModel()
summary(model)

train_loss_ls = []
val_loss_ls = []

time_stamp = str(time.time()).split(".")[0]
current_model_path = model_path + f"{model_type}-{time_stamp}-{n_epochs}e-{batch_size}b/"
os.mkdir(current_model_path)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr )
model.to(device)

last_weights, best_weights = train_vf(model, "baseline", batch_size, n_epochs, lr, trainroiloader,valroiloader, saved_model_path=current_model_path)