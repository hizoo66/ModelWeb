from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
from PIL import Image
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import io
import base64
from model import printImage
import matplotlib


matplotlib.use('Agg')


UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def SequentialImage(image):
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            order=[0, 1],
            cval=255
        )
    ], random_order=True)
    return seq.augment_image(image)

def preprocess_img(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((90, 90))
    img_array = np.array(image)
    
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    
    img_array = img_array / 255.0
    img_array = SequentialImage(img_array)
    img_array = np.reshape(img_array, (1, 90, 90, 1))
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

def predict_result(img, model):
    x_real = np.load('x_real.npz')['data']
    y_real = np.load('y_real.npy')
    result = None
    result_img = None
    for i in range(6010):
        real_x_i = x_real[i].reshape((90, 90, 1)).astype(np.float32) / 255.0
        prediction = model.predict([img, real_x_i.reshape((1, 90, 90, 1))])
        if prediction[0] == 1:
            result = prediction
            result_img = x_real[i]
            break
    return result, i, result_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('./fingerprintTest.h5')

@app.route("/")
def main():
    return render_template("index.html")

@app.route('/prediction', methods=['POST'])
def predict_image_file():
    if 'file' not in request.files:
        return "파일이 없습니다."
    
    f = request.files['file']
    filename = secure_filename(f.filename)
    upload_dir = f'{UPLOAD_FOLDER}/{filename}'
    f.save(upload_dir)
    
    img = preprocess_img(upload_dir)
    pred, num, _ = predict_result(img, model)
    x_real = np.load('x_real.npz')['data']
    y_real = np.load('y_real.npy')
    image, label = printImage(num, x_real, y_real)
    
    # Matplotlib 작업은 주 스레드에서 수행
    buffer = io.BytesIO()
    plt.figure(figsize=(2, 2))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    print(pred)
    return render_template("result.html", predictions=str(pred), image_path=upload_dir, img_str=img_str, label=label)

if __name__ == "__main__":
    app.run(port=9000, debug=True)