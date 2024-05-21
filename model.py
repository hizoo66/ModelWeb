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
model = load_model('./fingerprintTest.h5')

x_real = np.load('./x_real.npz')['data']
y_real = np.load('./y_real.npy')

def preprocess_img(img_path):
    image = Image.open(img_path).convert('L')  # 이미지 파일을 열고 그레이스케일로 변환
    image = image.resize((90, 90))  # 이미지 크기 조정
    img_array = np.array(image)  # 이미지를 numpy 배열로 변환
    
    # 배열이 2차원인지 확인하고 3차원으로 변환
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    
    img_array = img_array / 255.0  # 이미지를 0과 1 사이로 정규화
    img_array = SequentialImage(img_array)  # augmentation 적용
    img_array = np.reshape(img_array, (1, 90, 90, 1))  # 이미지 형태 재조정
    img_array = img_array.astype(np.float32) / 255.0  # 다시 0과 1 사이로 정규화
    return img_array

def predict_result(input_image, model):
    x_real = np.load('./x_real.npz')['data']
    input_image = input_image.reshape((1, 90, 90, 1)).astype(np.float32) / 255.0
    for i in range(6010):
        data_x = x_real[i].reshape((1, 90, 90, 1)).astype(np.float32) / 255.0
        pred = model.predict([input_image, data_x])
        if pred[0] == 1:
            return pred[0], i, input_image
        
    return None

    
def printImage(num, x_real, y_real):
    real = x_real[num].reshape((90, 90)).astype(np.float32) / 255.0  # 시각화를 위해 (90, 90)으로 reshape 조정
    label = y_real[num]
    info = [label[0]]
    
    info.append('Male' if label[1] == 0 else 'Female')
    info.append('Left' if label[2] == 0 else 'Right')
    
    finger_dict = {0: 'Thumb', 1: 'Index', 2: 'Middle', 3: 'Ring', 4: 'Little'}
    info.append(finger_dict.get(label[3], 'Unknown'))
    
    return real, info


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
