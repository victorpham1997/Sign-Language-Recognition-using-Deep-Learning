"""
Model Classes
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

VGG_MEAN = [103.939, 116.779, 123.68]

### FOR VALVE FILTER ###
class cnn_vf(nn.Module):
    def __init__(self):
        super(cnn_vf, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.conv1_0 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.vggmiddle = nn.Sequential(*list(vgg.features.children())[2:-5])
        
        for param in self.vggmiddle.parameters():
            param.requires_grad = False
        
        self.vggmiddle.add_module("conv5_2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.vggmiddle.add_module("relu5_2", nn.ReLU(inplace=True))
        self.vggmiddle.add_module("conv5_3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.vggmiddle.add_module("relu5_3", nn.ReLU(inplace=True))
        
    def forward(self, rgb, roi):
        '''
        Input shape = (batchsize, seq, 3, h, w)
        '''
#         print("Input rgb: ",rgb.shape)
#         print("Input roi: ",roi.shape)
        red,green,blue = rgb[:,0:1,:,:],rgb[:,1:2,:,:],rgb[:,2:,:,:]
#         print(red.shape,blue.shape,green.shape)
        bgr = torch.cat((blue - VGG_MEAN[0],green - VGG_MEAN[1], red - VGG_MEAN[2]),1)
#         print("BGR: ",bgr.shape)
        features_map = self.conv1_0(bgr)
#         print("features map: ",features_map.shape)
        relevance_map = self.conv1_1(roi)
#         print("relevance_map: ",relevance_map.shape)
        norm_features_map = features_map * relevance_map        
#         print("norm_features_map: ",norm_features_map.shape)
        output = self.vggmiddle(norm_features_map)
        return output
    



class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.pretrained_vgg = models.vgg16(pretrained=True)
        self.pretrained_vgg = nn.Sequential(*list(self.pretrained_vgg.features.children())[:-5])

        # Freeze parameter
        for param in self.pretrained_vgg.parameters():
            param.requires_grad = False

        # Manually add in the layers that are going to be finetuned
        self.pretrained_vgg.add_module("conv5_2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.pretrained_vgg.add_module("relu5_2", nn.ReLU(inplace=True))
        self.pretrained_vgg.add_module("conv5_3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.pretrained_vgg.add_module("relu5_3", nn.ReLU(inplace=True))
        
    def forward(self, x):
        '''
        Input shape = (batchsize, seq, 3, h, w)
        '''
        output = self.pretrained_vgg(x)
        return output
        
        
class fpm(nn.Module):
    def __init__(self):
        super(fpm, self).__init__()
        self.pool1_0 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1,dilation=2)
        self.conv1_1 = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=1)
        self.conv2_0 = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,padding=1)
        self.conv3_0 = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,padding=2,dilation=2)
        self.conv4_0 = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=3,padding=4,dilation=4)
        
    def forward(self, x):
        '''
        Input shape = (batchsize, seq, 512, 16, 16)
        '''
        x_1 = self.pool1_0(x)
        x_1 = F.relu(self.conv1_1(x_1))
        x_2 = F.relu(self.conv2_0(x))
        x_3 = F.relu(self.conv3_0(x))
        x_4 = F.relu(self.conv4_0(x))
        
        out = torch.cat([x_1,x_2,x_3,x_4], dim=1)
        
        return out

    
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional = True):
        super(lstm, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first = True , bidirectional=self.bidirectional)
    
    def forward(self, inputs, hidden):
        output, hidden_state = self.lstm(inputs, hidden)
        return output, hidden_state 
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1 + int(self.bidirectional), batch_size, self.hidden_size), torch.zeros(1 + int(self.bidirectional), batch_size, self.hidden_size))
    
    
class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.dropout_p = 0.25
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1, bias=False)
        
    def forward(self, x):
        hidden = x
        x = torch.tanh(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = F.softmax(x, dim=len(x.shape)-1)
        c = torch.sum(x*hidden, dim=1)
        return c
    
    

    
class ValveFilterModel(nn.Module):
    def __init__(self):
        super(ValveFilterModel, self).__init__()
        self.bidirectional = True
        self.lstm_hiddensize = 512
        self.dropout_p = 0.25
        
        self.cnn = cnn_vf()
        self.fpm = fpm()
        self.lstm = lstm(512, self.lstm_hiddensize, self.bidirectional)
        self.attn = attention()
        self.dropout1 = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(1024,226)
        
        
    def forward(self,x,roi):
        embedding_output = []
#         print("x: ",x.shape)
#         print("roi: ",roi.shape)
        for i in range(x.size(1)):
            x_i = self.cnn(x[:,i,:,:,:],roi[:,i,:,:,:])
            x_i = self.fpm(x_i)
            x_i = x_i.mean([-2, -1]) #Global Average Pooling
            embedding_output.append(x_i)
        embedding_output = torch.stack(embedding_output, dim=1)
        
        h_0, h_1 = self.lstm.init_hidden(embedding_output.shape[0])
        h_0 = h_0.to("cuda")
        h_1 = h_1.to("cuda")
        
        output, (final_hidden_state, final_cell_state) = self.lstm(embedding_output, (h_0,h_1)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        c = self.attn(output)
        output = self.dropout1(self.fc(c))
        output = F.log_softmax(output, dim=len(output.shape)-1)
        
        return output
    
### END OF VALVE FILTER ###