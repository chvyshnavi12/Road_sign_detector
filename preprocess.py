import cv2
import numpy as np
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.reshape(img, (1, 32, 32, 3))
    return img