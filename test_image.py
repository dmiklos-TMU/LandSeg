import torch
import numpy as np
import os
import torch
from model import LandSeg as net
import cv2

def Run(model,img_in):
    img, depth = img_in
    processedImg = img.copy()

    with torch.no_grad():
        img_tensor = torch.from_numpy(processedImg.transpose(2, 0, 1)).float().unsqueeze(0).cuda() / 255.0
        depth_tensor = torch.from_numpy(processedDepth).unsqueeze(0).unsqueeze(0).float().cuda() / 65535.0
        pred = model(img_tensor, depth_tensor)

    x0 = pred[0]
    _, landingZone= torch.max(x0, 0)
    landingZone = landingZone.byte().cpu().data.numpy()
        
    predMaskLZ = np.zeros_like(img, dtype=np.uint8)     
    predMaskLZ[landingZone ==1] = [255,255,255]
    predMaskLZ[landingZone ==0] = [0,0,0]
    
    return predMaskLZ

model = net.LandSeg()
model = model.cuda()
model.load_state_dict(torch.load(''))
model.eval()
image_path = ''
image_list = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
outputPath = ''

for i, imgName in enumerate(image_list):
    rgb = cv2.imread(os.path.join(image_path,imgName))
    depth = cv2.imread(os.path.join(image_path,imgName.replace(".jpg","-depth.png")), -1)
    predMask=Run(model,(rgb,depth))
    cv2.imwrite(os.path.join(outputPath,"predMask_"+imgName),predMask)
  
 
    
    
    
    
