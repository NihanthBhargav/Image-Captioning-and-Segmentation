import tensorflow as tf
import numpy as np
import os
import pickle

# Config
EMBEDDING_DIM = 256
UNITS = 512
VOCAB_SIZE = 5000
MAX_LEN = 20
MODEL_PATH = "models/caption_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Dummy Captions (replace later with COCO/Flickr8k)
captions = ["a cat on the bed", "a dog in the park", "a pet sitting"]

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(captions)
sequences = tokenizer.texts_to_sequences(captions)
padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

# Dummy image features (simulate CNN encoder output)
image_features = np.random.randn(len(captions), 2048)

# Model
# Image input
inputs_img = tf.keras.layers.Input(shape=(2048,))
img_dense = tf.keras.layers.Dense(EMBEDDING_DIM, activation="relu")(inputs_img)
img_repeat = tf.keras.layers.RepeatVector(MAX_LEN)(img_dense)

# Text input
inputs_text = tf.keras.layers.Input(shape=(MAX_LEN,))
emb = tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs_text)

# Merge image features and word embeddings
merged = tf.keras.layers.Concatenate()([img_repeat, emb])

# LSTM (return sequences → predict a word at each timestep)
lstm = tf.keras.layers.LSTM(UNITS, return_sequences=True)(merged)

# TimeDistributed Dense layer → word prediction for each timestep
outputs = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax")
)(lstm)

caption_model = tf.keras.Model([inputs_img, inputs_text], outputs)
caption_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Training
# Labels: same padded sequences (shape: num_samples, MAX_LEN)
labels = padded.astype(np.int32)

caption_model.fit(
    [image_features, padded],
    labels,
    epochs=2
)

# Save Model + Tokenizer
os.makedirs("models", exist_ok=True)
caption_model.save(MODEL_PATH)
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"Captioning model saved at {MODEL_PATH}")
print(f"Tokenizer saved at {TOKENIZER_PATH}")
