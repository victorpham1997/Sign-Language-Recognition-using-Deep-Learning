import torch
import torchvision
import numpy as np
import copy 

device = "cuda" if torch.cuda.is_available() else "cpu"
deeplabmodel = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).to(device).eval()

def segmentvideo(video):
    imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    preprocess = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean = imagenet_stats[0],
                                                                                std  = imagenet_stats[1])])

    def segmentimg(img, deeplabmodel):
        input_tensor = preprocess(img).unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():      
            output = deeplabmodel(input_tensor)["out"][0]
            output = output.argmax(0)
        return output

    results = []
    for i, frame in enumerate(video):  
        npframe = np.array(frame.permute(1,2,0))
        masks_tensor = segmentimg(npframe, deeplabmodel)
        results.append(masks_tensor)
        masks = torch.stack(results).cpu().numpy()
        maskbool = (masks==0)
        maskbool = np.repeat(maskbool[:, :, :, np.newaxis], 3, axis=3) # (batch, 512,512) mask to (batch, 512,512,3)
        maskedvideo = copy.deepcopy(video.permute(0,2,3,1)) #convert to b,w,h,c
        maskedvideo[maskbool] = 0 #set images outside mask to 0
