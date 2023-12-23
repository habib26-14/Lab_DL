# Lab11 : fashion_mnist
# Rèalisé par : Habib Tanou EMSI 2023 - 2024


# TensorFlow and tf.keras
import tensorflow as tf


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# Step 1: DataSet
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Data Preparation
train_images = train_images / 255.0
test_images = test_images / 255.0

# Step 2: Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Step 3: Train
model.fit(train_images, train_labels, epochs=50)

# Step 4: Test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nAccuracy test is: ', test_acc)

# Predict
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[0]

# Save the model
model.save("fashion_mnist_model.h5")
