from keras.models import load_model
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import os

import tensorflow as tf 
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Back End
model = load_model('./fingerprintTest1.h5')

x_real = np.load('./x_real.npz')['data']
"""
def extract_label(img_path):
    filename, _ = os.path.splitext(os.path.basename(img_path))
    
    subject_id, etc = filename.split('__')
    gender, lr, finger, _ = etc.split('_')
    
    gender = 0 if gender == 'M' else 1
    lr = 0 if lr =='Left' else 1
    
    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
        
    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)
"""
def preprocess_img(img_path):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((90, 90))
    img_array = np.array(img_resize)
    reshaped_data = img_array[:, :, :1]
    #img_reshape = np.expand_dims(reshaped_data, axis=0) #1 90 90 4
    #labels = extract_label(img_path)
    return reshaped_data

def predict_result(predict):
    x_real = np.load('./x_real.npz')['data']
    for i in range(100):
        predict = predict.reshape((1, 90, 90, 1)).astype(np.float32) / 255.
        data_x=x_real[i].reshape((1, 90, 90, 1)).astype(np.float32) / 255.
        pred = model.predict([predict, data_x])
        if pred[0]>0.9:
            return pred[0], data_x
        

def SequentialImage(image):          
    seq = iaa.Sequential([
        # blur images with a sigma of 0 to 0.5
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(
            # scale images to 90-110% of their size, individually per axis
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            # translate by -10 to +10 percent (per axis)
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            # rotate by -30 to +30 degrees
            rotate=(-30, 30),
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            # if mode is constant, use a cval between 0 and 255
            cval=255
        )
    ], random_order=True)

    return seq.augment_image(image)
