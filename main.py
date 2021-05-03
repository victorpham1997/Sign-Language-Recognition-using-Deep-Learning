import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.nn import functional as F

import argparse
import os
from utils.train import *
# from architecture.Baseline import Baseline, Transformer, ValveFilterModel
import cv2
import pandas as pd

import warnings 
warnings.filterwarnings("ignore")


def is_valid_file(parser, filepath, fileType):
	if os.path.exists(filepath):
		print ("Loading {} from: {}".format(fileType, filepath))
		return filepath
	else:
		raise parser.error("{} filepath: {} does not exist!".format(fileType, filepath))

parser = argparse.ArgumentParser(description='Valve Filter Model')
parser.add_argument('--model_name', type=str, required=True, help="Pass in the model type")
parser.add_argument('--model_weight', type=lambda x: is_valid_file(parser, x, "Model"), required=True, help="Pass in the model weight path")
parser.add_argument('--video', type=lambda x: is_valid_file(parser, x, "Video"), required=True, help="Pass in the video path")
parser.add_argument('--labels', type=lambda x: is_valid_file(parser, x, "Labels"), required=True, help="Pass in the ground truth labels")


args = parser.parse_args()
modelName = args.model_name
modelPath = args.model_weight
videoPath = args.video
labelsPath = args.labels


def displayPredVideo(videoPath, prediction):
	print ("Press 'q' to quit!")
	cap = cv2.VideoCapture(videoPath)
	font = cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 1
	thickness = 2
	color = (0, 0, 255)
	pred_string = "Predicted: {}".format(prediction.data.cpu().numpy()[0])
	while(True):
		ret, frame = cap.read()
		frame = cv2.putText(frame, pred_string, (25,50), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
		if ret:
			cv2.imshow("Image", frame)
		else:
			cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
				
	cap.release()
	cv2.destroyAllWindows()


def GetLabel(labelFilename, videoName):
	searchFor = "_".join(videoName.split("/")[-1].split("_")[:2])
	labels = pd.read_csv(labelFilename,names = ["file_name", "label"])
	return torch.tensor([labels[labels.file_name == searchFor]["label"].values[0]])


def GetVideoArray(videoFilePath,device):
	cap = cv2.VideoCapture(videoFilePath)   # capturing the video from the given path
	video_arr = []
	device = device
	while(cap.isOpened()):
		frameId = cap.get(1) #current frame number
		ret, frame = cap.read()
		if (ret != True):
			break
		if (frameId % 2 == 0):
			# Converting to tensor
			frame =  torchvision.transforms.functional.to_tensor(frame).float().to(device)
			frame = frame.unsqueeze(0)
			frame =  F.interpolate(frame, (256,256), mode='bilinear')
			frame = frame.squeeze(0)
			video_arr.append(frame)
	cap.release()

	return torch.unsqueeze(torch.stack(video_arr), 0) 


# load in video & roi data
device = "cuda"
videoArray = GetVideoArray(videoPath, device)
label = GetLabel(labelsPath, videoPath)


if modelName == "valveFilter":
	videoROIArray = GetVideoArray(videoPath.replace("color", "roi"),device)
	data = [videoArray, videoROIArray, label]
	from architecture.ValveFilterModel import *
	print("Asdfadsf")
elif modelName == "baseline":
	data = [videoArray, label]
	from architecture.BaselineModel import *
elif modelName == "transfromer":
	data = [videoArray, label]
	from architecture.TransformerModel import *


model = torch.load(modelPath)
model = model.to(device)
print (model)


predictions = inference(modelName, model, data, device)

displayPredVideo(videoPath, predictions)