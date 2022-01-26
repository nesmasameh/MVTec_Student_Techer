from Model.Prediction import predict
from Model.functions import data_transformation
import torch
from torchvision import models
test_transform = data_transformation()['test']


# print(save_path)
# # print(torchvision.__version__)

from flask import Flask,send_file,request
import cv2
import base64
from flask_cors import CORS 


app = Flask(__name__)
CORS(app)

@app.route("/<img_type>",methods=['GET','POST'])
def home(img_type):
    my_string = 'hello world'
    
    if 'media' in request.files:
        request.files['media'].save('img/1.jpg')
        img_copy = cv2.imread('img/1.jpg')
        # cv2.imwrite('result/1.jpg', img_copy) 
    print(img_type)    
    # img_path = r'E:\AI_Pro_intake1\graduation project\New folder\New folder\img\1.jpg'
    img_path = r'E:\AI_Pro_intake1\graduation project\final final\server\img\1.jpg'
    # img_type = 'carpet'
    save_path = predict(img_path, img_type, test_transform)
    with open(save_path, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
    
    return my_string

if __name__=='__main__':
    app.run(port=8080)