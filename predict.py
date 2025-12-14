import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("model/road_sign_cnn.h5")
class_names = np.load("model/classes.npy")

def predict_sign(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    return class_names[idx]
