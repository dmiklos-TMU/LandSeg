import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random

def blackout_RGB(img):
    height, width = img.shape[:2]
    black_img = np.zeros((height, width, 3), dtype=np.uint8)
    return black_img
    
def blackout_DEPTH(depth):
    height, width = depth.shape[:2]
    black_img = np.zeros((height, width), dtype=np.uint16)
    return black_img

class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, transform=None,valid=False):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''

        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid=valid
        if valid:
            self.root='./dataset/trainingSet1'
            self.names=[f for f in os.listdir(self.root) if f.endswith('.jpg')]
        else:
            self.root='./dataset/valSet2'
            self.names=[f for f in os.listdir(self.root) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_=640
        H_=480
        image_name=os.path.join(self.root,self.names[idx])
        
        image = cv2.imread(image_name)
        label1 = cv2.imread(image_name.replace(".jpg","-segLS.png"), 0)
        depth = cv2.imread(image_name.replace(".jpg","-depth.png"), -1)
     
        label1 = cv2.resize(label1, (W_, H_))
        image = cv2.resize(image, (W_, H_))
        depth = cv2.resize(depth, (W_, H_))
        
        rgbOut = False
        if random.random()<0.5:
            image=blackout_RGB(image)
            rgbOut = True
            
        if random.random()<0.5 and not rgbOut:
            depth=blackout_DEPTH(depth)


        _,seg_b1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY_INV)
        _,seg1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY)
        seg1 = self.Tensor(seg1)
        seg_b1 = self.Tensor(seg_b1)
        seg_LS = torch.stack((seg_b1[0], seg1[0]),0)

        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)        
        depth = np.ascontiguousarray(depth) 
        
        return image_name,(torch.from_numpy(image),torch.from_numpy(depth)),seg_LS
    



