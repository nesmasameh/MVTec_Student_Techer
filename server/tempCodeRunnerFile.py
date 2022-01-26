from logging import debug
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import cv2

from Model.Prediction import predict
from Model.functions import data_transformation
from Model.RequestPath import Image_path

test_transform = data_transformation()['test']

app = FastAPI()

@app.get('/')
def index() :
    return 'Anomaly Detection App'

@app.post('/predict') 
def predict_anomaly_map(img_data : Image_path) :
    data = img_data.dict()
    img_path = data['img_path']
    img_type = data['img_type']
    save_path = predict(img_path, img_type, test_transform)

    return FileResponse(save_path)


if __name__ == '__main__' :
    uvicorn.run(app, debug = True)



