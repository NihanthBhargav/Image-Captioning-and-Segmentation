import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Config
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
MODEL_PATH = "models/segmentation_model.h5"

# Preprocessing
def preprocess(image, mask):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    return image, mask

# Load dataset
dataset, info = tfds.load("oxford_iiit_pet:4.0.0", with_info=True)
train = dataset["train"].map(lambda x: preprocess(x["image"], x["segmentation_mask"])).batch(BATCH_SIZE)
test = dataset["test"].map(lambda x: preprocess(x["image"], x["segmentation_mask"])).batch(BATCH_SIZE)

# Build simple U-Net
def unet_model():
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

model = unet_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(train, validation_data=test, epochs=EPOCHS)

# Save model
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print(f"Segmentation model saved at {MODEL_PATH}")
