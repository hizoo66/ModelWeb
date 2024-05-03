from flask import Flask, render_template, request
from keras.models import load_model
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from model import preprocess_img, predict_result, SequentialImage
import numpy as np
from PIL import Image
import os
 
import tensorflow as tf 
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('./fingerprintTest1.h5')

@app.route("/")
def main():
    return render_template("index.html")
 
 
# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():

    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save('./static/uploads' + filename)
        img = preprocess_img('./static/uploads' + filename)
        pred , _= predict_result(img)
        return render_template("result.html", predictions=str(pred), filename = filename)
 
    """except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)"""
 
 
# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)