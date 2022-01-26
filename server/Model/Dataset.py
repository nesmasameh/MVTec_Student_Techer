import os 
import glob
import collections
import cv2
import numpy as np
from torch.utils.data import Dataset


class MVTec_Dataset(Dataset) :
    def __init__(self, data_path, category, transforms = None, mode = 'train') :
        self.data_path = os.path.join(data_path, category)
        self.mode = mode
        self.transforms = transforms
        self.img_info = self.get_data()
        
    
    def __len__(self) :
        return len(self.img_info)
    
    def __getitem__(self, idx) :
        img_path = self.img_info[idx]['img_path']
        mask_path = self.img_info[idx]['mask_path']
        label = self.img_info[idx]['label']
        typ = self.img_info[idx]['type']
        
        # get the image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask_path  :
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (256, 256))
            #mask = mask / 255
            
        else :
            mask = np.zeros((image.shape[0], image.shape[1]))
            mask = cv2.resize(mask, (256, 256))
        if self.transforms :
            image = self.transforms(image)
#             mask = self.transforms(mask)
        return image, mask, label, typ
    
    def visualize(self) :
        pass 
    
    def get_data(self) :
        
        #classes = os.listdir(self.data_path)
        img_info = collections.defaultdict()
        idx = 0
        #for cls in classes :
        if self.mode == 'train' :
            path = os.path.join(self.data_path, 'train')
        else :
            path = os.path.join(self.data_path, 'test')
            gt_path = os.path.join(self.data_path, 'ground_truth')
        defect_types = os.listdir(path)
        for typ in defect_types :
            if typ == 'good' :
                images_path = glob.glob(os.path.join(path, typ) + '/*.png')
                defect_types = typ
                image_label = 0
                masks_path = None 

                for i in range(len(images_path)) :
                    img_info[idx] = {'img_path' : images_path[i], 'mask_path' : masks_path
                                     ,'label' : image_label, 'type' : defect_types}
                    idx += 1  
            else :

                images_path = glob.glob(os.path.join(path, typ) + '/*.png') 
                defect_types = typ
                image_label = 1
                masks_path = glob.glob(os.path.join(gt_path, typ) + '/*.png')

                for i in range(len(images_path)) :
                    img_info[idx] = {'img_path' : images_path[i], 'mask_path' : masks_path[i]
                                     ,'label' : image_label, 'type' : defect_types}
                    idx += 1

        return img_info
    
    
