# Lab21 : Classification des fruits et legumes
# Realise par Habib Tanou EMSI 2023/2024
# Reference : https://colab.research.google.com/drive/1MY9rqisajfcCvS3kGCnJuoA_wb549bNU#scrollTo=8lCi3O2Otp9D

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy


# Step 1 : Dataset
img_height, img_width = 32, 32
batch_size = 20

train_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/train",
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/validation",
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "datasets/fruits/test",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Data Visualisation
class_names = ["apple", "banana", "orange"]
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Step 2: Model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(3)
    ]
)

# Compile
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Step 3: Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

# Step 4: Test
model.evaluate(test_ds)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    classifications = model(images)

    # Print(classifications)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        index = numpy.argmax(classifications[i])
        plt.title("Pred: " + class_names[index] + " | Real: " + class_names[labels[i]])

# Save the model as TfLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", 'wb') as f:
    f.write(tflite_model)
