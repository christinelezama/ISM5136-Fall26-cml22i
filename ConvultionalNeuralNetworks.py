#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:55:19 2026

@author: christinelezama
"""

#1.1
import pandas as pd
from pathlib import Path

downloads = Path.home() / "Downloads"

pokemon = pd.read_csv(downloads / "pokemon_all (2).csv")
pokemon = pokemon.dropna(subset=["Type_2"])

print(pokemon.shape)
print(pokemon[["Number", "Name", "Type_2"]].head())


#1.2
from zipfile import ZipFile

zip_path = downloads / "pokemon_png (1).zip"

with ZipFile(zip_path) as zip_file:
    zip_file.extractall(downloads)

image_folder = downloads / "pokemon_png"
print("PNG files:", len(list(image_folder.glob("*.png"))))

import numpy as np
from PIL import Image

images = []
matched_rows = []

for index, row in pokemon.iterrows():
    image_path = image_folder / f"{row['Number']}.png"

    try:
        with Image.open(image_path) as picture:
            gray_picture = picture.convert("L")
            images.append(np.array(gray_picture))
        matched_rows.append(index)
    except FileNotFoundError:
        pass

X = np.array(images)
pokemon = pokemon.loc[matched_rows].copy()

print("Image array:", X.shape)
print("Matching Pokémon:", len(pokemon))


#1.3
X = X.astype("float32") / (X.max() - X.min())

print("Smallest value:", X.min())
print("Largest value:", X.max())
print("Question 1:", X[0, 70, 35])



#1.4
y_table = pd.get_dummies(pokemon["Type_2"], dtype=int)

class_names = y_table.columns.tolist()
y = y_table.to_numpy()

print("Question 2 — shape of y:", y.shape)
print("Type_2 classes:", class_names)


#2
import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.Input(shape=(256, 256, 1)))

model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(y.shape[1], activation="softmax"))


model.summary()

X_model = X.reshape(X.shape[0], 256, 256, 1)
print("Shape for TensorFlow:", X_model.shape)


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(X_model, y, epochs=10, batch_size=16)

print("Question 4 — final accuracy:", history.history["accuracy"][-1])

#3
print(list(downloads.glob("*predict*")))

predict_path = downloads / "Pokemon_predict_image.png"

with Image.open(predict_path) as picture:
    gray_picture = picture.convert("L")
    predict_array = np.array(gray_picture)

print(predict_array.shape)

predict_array = predict_array.astype("float32") / 255.0
predict_array = predict_array.reshape(1, 256, 256, 1)

print(predict_array.shape)

probabilities = model.predict(predict_array)[0]

best = np.argmax(probabilities)

print("Question 5 — most likely class:", class_names[best])
print("Probability:", round(probabilities[best] * 100, 2), "%")

print(list(downloads.glob("*.png")))

print("\n===== POKÉMON CNN LAB ANSWERS =====")

print("\nQuestion 1 — normalized pixel value:")
print(X[0, 70, 35])

print("\nQuestion 2 — shape of y:")
print(y.shape)

print("\nQuestion 3 — model architecture:")
model.summary()

print("\nQuestion 4 — final training accuracy:")
print(f"{history.history['accuracy'][-1]:.2%}")

print("\nQuestion 5 — predicted Type_2 and probability:")
print(class_names[best])
print(f"{probabilities[best]:.2%}")
