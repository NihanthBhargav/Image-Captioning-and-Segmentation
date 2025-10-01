import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pickle

IMG_SIZE = 128
MAX_LEN = 20

# Load models + tokenizer
segmentation_model = tf.keras.models.load_model("models/segmentation_model.h5")
caption_model = tf.keras.models.load_model("models/caption_model.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocess
def preprocess(sample):
    img = tf.image.resize(sample["image"], (IMG_SIZE, IMG_SIZE)) / 255.0
    mask = tf.image.resize(sample["segmentation_mask"], (IMG_SIZE, IMG_SIZE), method="nearest")
    return img, mask

# Dataset
dataset, info = tfds.load("oxford_iiit_pet:4.0.0", with_info=True)
test = dataset["test"].map(preprocess).batch(1)

# Caption generator (dummy greedy decoding)
def generate_caption(model, tokenizer, photo_features, max_len=MAX_LEN):
    return "Generated caption for pet image"

# Run pipeline
for img, mask in test.take(1):
    pred_mask = segmentation_model.predict(img)[0]
    caption = generate_caption(caption_model, tokenizer, img)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img[0])
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Caption: {caption}")
    plt.imshow(img[0])
    plt.axis("off")

    plt.show()
