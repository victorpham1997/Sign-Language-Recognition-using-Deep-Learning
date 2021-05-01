import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.nn import functional as F
from torchsummary import summary
from glob import glob

import argparse
import os
from ModelClasses import *
import cv2
import pandas as pd

import warnings 
warnings.filterwarnings("ignore")

class AUTSLDatasetROI(Dataset):
    '''
    Dataset for the AUTSL data
    '''
    
    
    def __init__(self, data_type, max_frame_no ,frame_interval = 1, file_percentage = 1.0, data_path ="./dataset/" ):
        self.data_type = data_type
        self.frame_interval = frame_interval
        self.data_path = {"train": data_path + "train/",
                          "val": data_path + "val/"
                         }        
        self.data_length = {"train":  int(len(glob(f"{self.data_path['train']}*_color.mp4"))*file_percentage),
                            "val": int(len(glob(f"{self.data_path['val']}*_color.mp4"))*file_percentage)
                           }
        # self.data_length = {"train":  len(os.listdir(self.data_path["train"]))}
        self.label_path = {"train": data_path + "train_labels.csv",
                           "val": data_path + "val_labels.csv"
                          }
        self.labels = pd.read_csv(self.label_path[self.data_type],names = ["file_name", "label"])
#         self.max = 156
        self.file_ls = glob(f"{self.data_path[self.data_type]}*_color.mp4")[:self.data_length[self.data_type]]
        self.max_frame_no = max_frame_no
        self.device = "cuda"

#         self.max = self.get_max()
        
    def Describe(self):
        msg = "AUTSL Dataset\n"
        print(msg)
    
#     def CreateLabel(self):
#         label_df = pd.read_csv(self.label_path[self.data_type],names = ["file_name", "label"])
#         label_df["signer"] = label_df.apply(lambda r: int(r.file_name.split("_")[0][6:]), axis =1)
#         return label_df   
        
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
        return self.data_length[self.data_type]
    
    def __getitem__(self, index):
        '''
        Return 4D array consists of all the frame of the video image AND the label
        '''
        file_name = os.path.basename(self.file_ls[index])
        d_file_name = file_name.replace("color","roi")
        
        video_arr = self.GetVideoArray(file_name)
        label = self.GetLabel(file_name[:-10]) #slice to get just the name without file ext and file type
        videoroi_arr = self.GetVideoArray(file_name.replace("color","roi"))
        file_name.replace("roi","color")
        return video_arr, videoroi_arr, label
        
    def get_max(self):
        max_length = 0
        for i in tqdm(range(len(self))):
            video, label = self[i]
            if video.shape[0]>max_length:
                max_length = video.shape[0]
        return max_length
    
torch.manual_seed(0)

batch_size = 2

trainROI_autsl = AUTSLDatasetROI("train", int(156/2), frame_interval=4, file_percentage=0.3)
valROI_autsl = AUTSLDatasetROI("val", int(114/2), frame_interval=4, file_percentage=0.3)
trainroiloader = DataLoader(trainROI_autsl, batch_size=batch_size, shuffle=True)
valroiloader = DataLoader(valROI_autsl, batch_size=batch_size, shuffle=True)

## Pre-processing
TRAIN_PATH = "../train/"
VAL_PATH = "../val/"
files_train = os.listdir(TRAIN_PATH)
filtered_train = [file for file in files_train if "depth" in file]
files_val = os.listdir(VAL_PATH)
filtered_val = [file for file in files_val if "depth" in file]

def generate_roi(filter_path, filtered):
    for filename in tqdm(filtered):
        name = filter_path + filename 
    #     name = target1
        cap = cv2.VideoCapture(name)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        filename = name.replace("depth","roi")
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#         print(filename)
        out = cv2.VideoWriter(filename, fourcc, 30, (512, 512))

        while True:
            _, frame = cap.read()
            if frame is None:
                break
            output = frame.copy()
            retval,thresh = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5),np.uint8)
            dilation = cv2.dilate(thresh,kernel,iterations = 1)
            out.write(dilation)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        out.release()
		
generate_roi(TRAIN_PATH, filtered_train)
generate_roi(VAL_PATH, filtered_val)


# Define model instance 
model = BaselineModel()
# x = torch.rand((5, 4, 3, 256, 256))
summary(model)

# Define Parameter here:
model_type = "baseline"
model_path = "./model/"
n_epochs = 5
lr = 1e-5
steps = 0 
print_every = 50
validate_every = 50
device = "cuda"

train_loss_ls = []
val_loss_ls = []


time_stamp = str(time.time()).split(".")[0]
current_model_path = model_path + f"{model_type}-{time_stamp}-{n_epochs}e-{batch_size}b/"
os.mkdir(current_model_path)


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr )
model.to(device)

# Model Functions (Train, Val, Test)
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
#     confusion_matrix = torch.zeros(num_classes, num_classes)
    with tqdm(testloader, position=0, leave=False) as progress_bar:          
        for images, rois, labels in progress_bar:
    #     for images, labels in testloader:
            images, rois, labels = images.to(device), rois.to(device), labels.to(device)

            output = model(images, rois)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            predictions = ps.max(dim=1)[1]
            equality = (labels.data == predictions)
            accuracy += equality.type(torch.FloatTensor).mean()
    #         for label, prediction in zip(labels.view(-1), predictions.view(-1)):
    #             confusion_matrix[label.long(), prediction.long()] += 1
    return test_loss, accuracy


def test(model, testloader, device='cuda'):  
    model.to(device)
    accuracy = 0

    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
                    
            output = model(images)
            
            ps = torch.exp(output)
            predictions = ps.max(dim=1)[1]
            equality = (labels.data == predictions)
            accuracy += equality.type(torch.FloatTensor).mean()

            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
#         recall = metrics.recall(confusion_matrix, num_classes)
#         precision = metrics.precision(confusion_matrix, num_classes)
#         f1 = metrics.f1(confusion_matrix, num_classes)
        print('Testing accuracy: {:.3f}'.format(accuracy/len(testloader)))
        print(f'Testing recall: {recall:.3f}')
        print(f'Testing precision: {precision:.3f}')
        print(f'Testing f1: {f1:.3f}')

    return accuracy, confusion_matrix


def train(model, model_name, batch_size, n_epochs, lr, trainroi_loader, valroi_loader, saved_model_path, device = "cuda"):
#     input_sample, _ =  next(iter(train_loader))
#     print(summary(model, tuple(input_sample.shape[1:]), device=device))
    start_time = datetime.now()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr= lr)
#     if use_lr_scheduler:
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)

    train_loss_ls = []
    val_loss_ls = []

    best_accuracy = 0
    best_recall = 0
    best_accuracy_weights = None
    best_recall_weights = None
    
    steps = 0
    
    running_loss = 0.0
    running_loss2 = 0.0
    
    for e in range(n_epochs):  # loop over the dataset multiple times

        # Training
        model.train()

#         with tqdm(trainroi_loader, position=0, leave=False) as progress_bar:  
#             for images, roi, labels in progress_bar:
        train_it= iter(trainroi_loader)
        for it in tqdm(range(len(trainroi_loader))):
            images,roi,labels = next(train_it)
    
#             print(images.shape)
#             print(roi.shape)
#             print(labels.shape)
            steps += 1
#                 images, labels = images.to(device), labels.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(images,roi)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss2 += loss.item()

            if steps % validate_every == -1:

                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, val_loader, criterion, device)
#                         recall = metrics.recall(confusion_matrix, num_classes)
#                         precision = metrics.precision(confusion_matrix, num_classes)
#                         f1 = metrics.f1(confusion_matrix, num_classes)

#                     filepath = saved_model_path + f"{model_name}-{start_time}-b{batch_size}-e{e}.pt"
#                     torch.save(model, filepath)

                running_loss /= validate_every

                time_elapsed = (datetime.now() - start_time)
                tqdm.write(f'===Epoch: {e+1}===')
                tqdm.write(f'== Loss: {running_loss:.3f} Time: {datetime.now()} Elapsed: {time_elapsed}')    
                tqdm.write(f'== Val Loss: {test_loss/len(val_loader):.3f} Val Accuracy: {accuracy/len(val_loader):.3f}') 
#                     tqdm.write(f'== Val Recall: {recall:.3f} Val Precision: {precision:.3f} Val F1: {f1:.3f}')

#         if recall > best_recall:
#             best_recall_weights = model.state_dict()
#             best_recall = recall
#             tqdm.write(f'\n=== BEST RECALL!!! ===')

                if accuracy > best_accuracy:
                    best_accuracy_weights = model.state_dict()
                    best_accuracy = accuracy
                    tqdm.write(f'\n=== BEST ACCURACY!!! ===')

                train_loss_ls.append(running_loss) #/print_every
                val_loss_ls.append(test_loss/len(val_loader))
                running_loss = 0        

                # Make sure training is back on
                model.train()
            elif  steps % print_every == 0:
                print("Epoch: {}/{} - ".format(e+1, n_epochs), "Training Loss: {:.3f} - ".format(running_loss2/print_every))
                running_loss2 = 0
                    
        filepath = saved_model_path + f"{model_name}-{start_time}-b{batch_size}-e{e}.pt"
        torch.save(model, filepath)

    print("Finished training")
    
    plt.plot(train_loss_ls, label = "train_loss")
    plt.plot(val_loss_ls, label = "val_loss")
    plt.legend()
    plt.savefig(saved_model_path+'train_val_loss.png')
    plt.show()
    return model.state_dict(), best_accuracy_weights

last_weights, best_weights = train(model, "baseline", batch_size, n_epochs, lr, trainroiloader,valroiloader, saved_model_path=current_model_path)

model.eval()

# Turn off gradients for validation
with torch.no_grad():
    test_loss, accuracy = validation(model, valloader, criterion, device)
#                         recall = metrics.recall(confusion_matrix, num_classes