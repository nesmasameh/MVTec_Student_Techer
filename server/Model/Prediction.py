import cv2
import os
import torch
from Model.Inference import Inference
from Model.functions import heatmap_on_image, min_max_norm, cvt2heatmap
# from functions import data_transformation

# all_chk_paths = r'E:\AI_Pro_intake1\graduation project\New folder\New folder\Model State Dicts'

all_chk_paths = r'E:\AI_Pro_intake1\graduation project\final final\server\Model State Dicts'
def predict(img_path, img_type, transform):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = transform(image)
    image = image.unsqueeze(axis = 0)
    model_chk_path = os.path.join(all_chk_paths, f'model_s_{img_type}.pt')
    weights = torch.load(model_chk_path, map_location=torch.device('cpu'))
    model=Inference(weights)
    a_maps, anomaly_map = model.anomaly_map(image)
    
    img = image.squeeze().numpy().transpose(1,2,0)   
    anomaly_map = anomaly_map.detach().squeeze().numpy()
    
    heatmap = cvt2heatmap(min_max_norm(anomaly_map)*255)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    hm_on_img = heatmap_on_image(heatmap, img*255)
    
    os.makedirs(os.path.join(os.getcwd(),'results'), exist_ok=True)
    img_name = img_path.split('\\')[-1]
    save_path = os.path.join(os.getcwd(),'results', f'mask{img_name}')
    cv2.imwrite(save_path, hm_on_img*255)


    return save_path

# test_transform = data_transformation()['test']
# img_path = r'D:\ITI-- -AI-PRO\competetions\GP Anomaly Detection\MVTec Anomaly Detection Dataset\carpet\test\color\014.png'
# img_type = 'carpet'

# save_path = predict(img_path, img_type, test_transform)

# print(save_path)