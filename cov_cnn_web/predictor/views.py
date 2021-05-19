import os
import cv2
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from keras.models import model_from_json
from keras.preprocessing import image
from django.core.files.storage import FileSystemStorage


# Create your views here.

covid_pred = ['Covid-19', 'Non Covid-19']
IMAGE_SIZE = 64
vgg16_model = 'predictor/model_weights/VGG16/VGG16_Model.hdf5'
vgg16_json = 'predictor/model_weights/VGG16/VGG16_Model.json'
resnet_model = 'predictor/model_weights/ResNet50/ResNet50_Model.hdf5'
resnet_json = 'predictor/model_weights/ResNet50/ResNet50_Model.json'
xception_model = 'predictor/model_weights/Xception/Xception_Model.hdf5'
xception_json = 'predictor/model_weights/Xception/Xception_Model.json'

def read_image(filepath):
    return cv2.imread(filepath) 

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

def clear_mediadir():
    media_dir = "./media"
    for f in os.listdir(media_dir):
        os.remove(os.path.join(media_dir, f))

def index(request):
    if request.method == "POST" :
        clear_mediadir() 
        
        img = request.FILES['ImgFile']

        fs = FileSystemStorage()
        filename = fs.save(img.name, img)
        img_path = fs.path(filename)

        pred_arr = np.zeros(
            (1, IMAGE_SIZE, IMAGE_SIZE, 3))

        im = read_image(img_path)
        if im is not None:
            pred_arr[0] = resize_image(im, (IMAGE_SIZE, IMAGE_SIZE))
        
        pred_arr = pred_arr/255
        
        vgg_start = time.time()
        with open(vgg16_json, 'r') as vggjson:
            vgg16model = model_from_json(vggjson.read())

        vgg16model.load_weights(vgg16_model)
        label_vgg = vgg16model.predict(pred_arr)
        idx_vgg = np.argmax(label_vgg[0])
        cf_score_vgg = np.amax(label_vgg[0])
        vgg_end = time.time()

        vgg_exec = vgg_end  - vgg_start

        resnet_start = time.time()
        with open(resnet_json, 'r') as resnetjson:
            resnetmodel = model_from_json(resnetjson.read())

        resnetmodel.load_weights(resnet_model)
        label_resnet = resnetmodel.predict(pred_arr)
        idx_resnet = np.argmax(label_resnet[0])
        cf_score_resnet = np.amax(label_resnet[0])
        resnet_end = time.time()

        resnet_exec = resnet_end - resnet_start

        xception_start = time.time()
        with open(xception_json, 'r') as xceptionjson:
            xceptionmodel = model_from_json(xceptionjson.read())

        xceptionmodel.load_weights(xception_model)
        label_xception = xceptionmodel.predict(pred_arr)
        idx_xception = np.argmax(label_xception[0])
        cf_score_xception = np.amax(label_xception[0])
        xception_end = time.time()

        xception_exec = xception_end - xception_start

        print('Prediction (VGG16): ', covid_pred[idx_vgg])
        print('Confidence Score (VGG16) : ', cf_score_vgg)
        print('Prediction Time (VGG) : ', vgg_exec)
        print("\n")
        print('Prediction (ResNet50): ', covid_pred[idx_resnet])
        print('Confidence Score (ResNet50) : ',cf_score_resnet)
        print('Prediction Time (ResNet50) : ', resnet_exec)
        print("\n")
        print('Prediction (Xception): ', covid_pred[idx_xception])
        print('Confidence Score (Xception) : ', cf_score_xception)
        print('Prediction Time (Xception) : ', xception_exec)
        print("\n")
        print(img_path)

        response = {}
        response['table'] = "table"
        response['col0'] = " "
        response['col1'] = "VGG16"
        response['col2'] = "ResNet50"
        response['col3'] = "Xception"
        response['row1'] = "Results"
        response['row2'] = "Confidence Score"
        response['row3'] = "Prediction Time (s)"
        response['r_pred'] = covid_pred[idx_resnet]
        response['v_pred'] = covid_pred[idx_vgg]
        response['x_pred'] = covid_pred[idx_xception]
        response['r_cf'] = cf_score_resnet
        response['v_cf'] = cf_score_vgg
        response['x_cf'] = cf_score_xception 
        response['r_time'] = resnet_exec
        response['v_time'] = vgg_exec
        response['x_time'] = xception_exec 
        response['image'] = "../media/" + img.name
        return render(request, 'index.html', response)
    else:
        return render(request, 'index.html')
