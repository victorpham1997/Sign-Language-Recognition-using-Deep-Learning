import torch
import torchvision
import numpy as np
import copy 

def segment_rm_bg(video, segmentation_model, device):
    """
    input video is a tensor
    output video is a tensor
    """
    imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                std  = imagenet_stats[1])])

    def segmentimg(img):
        input_tensor = preprocess(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():      
            output = segmentation_model(input_tensor)["out"][0]
            output = output.argmax(0)
        return output

    masks = []
    for i, frame in enumerate(video):  
        npframe = np.array(frame.permute(1,2,0))
        masks_tensor = segmentimg(npframe)
        masks.append(masks_tensor)
    
    masks = torch.stack(masks).cpu().numpy()
    maskbool = (masks==0)
    maskbool = np.repeat(maskbool[:, :, :, np.newaxis], 3, axis=3) # (batch, 512,512) mask to (batch, 512,512,3)
    maskedvideo = copy.deepcopy(video.permute(0,2,3,1)) #convert to b,w,h,c
    maskedvideo[maskbool] = 0 #set images outside mask to 0
    return torch.tensor(maskedvideo)


def segment_img_rm_bg(frame, segmentation_model, device):
    """
    input video is a tensor
    output video is a tensor
    """
    imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                std  = imagenet_stats[1])])

    def segmentimg(img):
        input_tensor = preprocess(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():      
            output = segmentation_model(input_tensor)["out"][0]
            output = output.argmax(0)
        return output
    
    npframe = frame.permute(1,2,0).cpu().numpy()
    mask = segmentimg(npframe).cpu().numpy()
    # print(np.unique(mask))
    maskbool = (mask!=15)
    maskbool = np.repeat(maskbool[:, :, np.newaxis], 3, axis=2) # (512,512) mask to (512,512,3)
    npframe[maskbool] = 0
    return npframe
