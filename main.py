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
from utils.AUTSLDataset import AUTSLDataset 

t  = AUTSLDataset("train")
max_frame = t.get_max()
print(max_frame)