import os 
import glob
from torch.utils.data import Dataset, DataLoader 
import collections
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def data_transformation() :
    
    transform = {'train' : transforms.Compose([transforms.ToPILImage(),
                                               transforms.Resize((256, 256)),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomRotation(45),
                                               transforms.RandomVerticalFlip(p=0.5),
                                               transforms.ToTensor(),
                                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                              ]), 

                 'test' : transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((256, 256)),
                                              transforms.ToTensor(),
                                              #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                              ])}
    return transform



def cost_function(F_t, F_s) :
    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0
    for key in F_s.keys() :
        f_s = F.normalize(F_s[key], p = 2)
        f_t = F.normalize(F_t[key], p = 2)
        _, _, h, w = f_s.shape
        loss = (1/(2*w*h)) * criterion(f_t, f_s)
        total_loss += loss
    return total_loss  


def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    out = np.float32(image)/255 + np.float32(heatmap)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    




def model_train(epochs, batch_size, train_set, valid_set, model, learn_rate) :
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
    validation_loader = DataLoader(valid_set, batch_size = batch_size, shuffle=True)
    
    optimizer = optim.SGD(params=model.student_model.parameters(), lr = learn_rate)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    valid_loss_min = np.Inf
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        train_loss = 0.0
        valid_loss = 0.0
        
        model.train()
        for batch_idx, data in enumerate(train_loader, 0):
          # get the inputs; data is a list of [inputs, labels]
            image, mask, label, typ = data
            image, mask = image.to(device), mask.to(device)

          # zero the parameter gradients
            optimizer.zero_grad()

          # forward + backward + optimize

            features_t = model.teacher(image)
            output = model(image)

            loss = cost_function(features_t, output)

            loss.backward()
            optimizer.step()

          # print statistics
            train_loss += loss.item() * image.size(0)
          #print(running_loss)]    

        model.eval()
        for batch_idx, data in enumerate(validation_loader, 0) :
            # move tensors to GPU if CUDA is available
            image, mask, label, typ = data
            image, mask = image.to(device), mask.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            features_t = model.teacher(image)
            output = model(image)
            # calculate the batch loss
            loss = cost_function(features_t, output)
            # update average validation loss 
            valid_loss += loss.item() * image.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(validation_loader.dataset)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
        # save model if validation loss has decreased
        if epoch > 80 : 
            
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.student_model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss

        
    print('Finished Training') 
