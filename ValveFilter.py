import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.nn import functional as F

import argparse
import os
from ModelClasses import *
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
parser.add_argument('--model', type=lambda x: is_valid_file(parser, x, "Model"), required=True, help="Pass in the model path")
parser.add_argument('--video', type=lambda x: is_valid_file(parser, x, "Video"), required=True, help="Pass in the video path")
parser.add_argument('--labels', type=lambda x: is_valid_file(parser, x, "Labels"), required=True, help="Pass in the ground truth labels")


args = parser.parse_args()
modelPath = args.model
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
	searchFor = "_".join(videoName.split("\\")[-1].split("_")[:2])
	labels = pd.read_csv(labelFilename,names = ["file_name", "label"])
	return torch.tensor([labels[labels.file_name == searchFor]["label"].values[0]])


def GetVideoArray(videoFilePath):
	cap = cv2.VideoCapture(videoFilePath)   # capturing the video from the given path
	video_arr = []
	device = "cuda"
	while(cap.isOpened()):
		frameId = cap.get(1) #current frame number
		ret, frame = cap.read()
		if (ret != True):
			break
		if (frameId % 4 == 0):
			# Converting to tensor
			frame =  torchvision.transforms.functional.to_tensor(frame).float().to(device)
			frame = frame.unsqueeze(0)
			frame =  F.interpolate(frame, (128,128), mode='bilinear')
			frame = frame.squeeze(0)
			video_arr.append(frame)
	cap.release()


	if len(video_arr)<int(156/2):
		empty_frame = torch.zeros((3,128,128)).to(device)
		padding  = [empty_frame]*(int(156/2)-len(video_arr))
		video_arr+=padding

	return video_arr, torch.unsqueeze(torch.stack(video_arr), 0) 

	
# load in video & roi data
videoArrayRaw, videoArray = GetVideoArray(videoPath)
videoROIArrayRaw, videoROIArray = GetVideoArray(videoPath.replace("color", "roi"))

# load in the gt labels
label = GetLabel(labelsPath, videoPath)

# model = BaselineModel()
# print (model)
model = torch.load(modelPath)
print (model)
criterion = nn.NLLLoss()
predictions, test_loss, accuracy = validation_vf(model, [videoArray, videoROIArray, label], criterion, "cuda")
displayPredVideo(videoPath, predictions)