import numpy as np
from keras.models import load_model
from PIL import Image
import imgaug.augmenters as iaa

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
    img_array = img_array.astype(np.float32)
    return img_array

def predict_result(img, model):
    x_real = np.load('x_real.npz')['data']
    y_real = np.load('y_real.npy')
    result = None
    result_img = None
    result_index = None

    # 디버깅을 위한 로그 출력
    print("Input image shape:", img.shape)
    
    for i in range(100):
        real_x_i = x_real[i].reshape((90, 90, 1)).astype(np.float32) / 255.0
        prediction = model.predict([img, real_x_i.reshape((1, 90, 90, 1))])**(10**29)
        
        # 예측 값이 9보다 큰 경우 출력
        if prediction[0] > 0:
            print(f"Iteration {i}: prediction={prediction[0]}, real_x_i shape={real_x_i.shape}")
        
        if prediction[0] > 0.001:
            result = prediction
            result_img = x_real[i]
            result_index = i
            break

    # 예측 결과가 없는 경우를 처리
    if result is None:
        print("No prediction with value 1 found.")
        return None, None, None
    
    return result, result_index, result_img

def printImage(num, x_real, y_real):
    real = x_real[num].reshape((90, 90)).astype(np.float32) / 255.0  # 시각화를 위해 (90, 90)으로 reshape 조정
    label = y_real[num]
    info = [label[0]]
    
    info.append('Male' if label[1] == 0 else 'Female')
    info.append('Left' if label[2] == 0 else 'Right')
    
    finger_dict = {0: 'Thumb', 1: 'Index', 2: 'Middle', 3: 'Ring', 4: 'Little'}
    info.append(finger_dict.get(label[3], 'Unknown'))
    
    return real, info


if __name__ == "__main__":
    # 모델을 로드하고 사용할 준비가 되었는지 확인합니다.
    model = load_model('./fingerprintTest.h5')

    # 예제 이미지 경로
    image_path = './2__F_Right_ring_finger.bmp'
    img = preprocess_img(image_path)
    pred, num, result_img = predict_result(img, model)
    if num is not None:
        x_real = np.load('x_real.npz')['data']
        y_real = np.load('y_real.npy')
        image, label = printImage(num, x_real, y_real)
        
    
    # 결과 출력
    print("Prediction:", pred)
    print("Index:", num)
    print("Label Info:", label)
    if result_img is not None:
            print("Result Image:", result_img.shape)
    else:
            print("Result Image: None")
else:
    print("No valid prediction found.")