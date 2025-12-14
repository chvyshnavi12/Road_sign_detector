import tensorflow as tf
import numpy as np
import os

# --------------------
# KERAS ALIASES (AS REQUESTED)
# --------------------
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# --------------------
# CONFIG
# --------------------
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 25

TRAIN_DIR = "dataset/train"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "road_sign_cnn.h5")
CLASS_PATH = os.path.join(MODEL_DIR, "classes.npy")

# Create model directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------
# DATA GENERATORS
# --------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

# Save class names
class_names = list(train_gen.class_indices.keys())
np.save(CLASS_PATH, class_names)

# --------------------
# CNN MODEL
# --------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

# --------------------
# COMPILE
# --------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------
# TRAIN
# --------------------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# --------------------
# SAVE MODEL
# --------------------
model.save(MODEL_PATH)
print("✅ Model saved at:", MODEL_PATH)
print("✅ Classes saved at:", CLASS_PATH)
