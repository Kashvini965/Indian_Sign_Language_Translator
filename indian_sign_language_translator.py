

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

dataset_path = "/content/drive/MyDrive/Indian New Data/archive (1)"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

class_labels = list(train_data.class_indices.keys())
print("Classes:", class_labels)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save trained model
model.save("/content/ISL_Translator.h5")
print("Model saved!")

# If your dataset is only alphabets A-Z
class_labels = [chr(i) for i in range(65, 91)]   # ['A','B',...,'Z']

# If your dataset has A-Z + digits 0-9
# class_labels = [chr(i) for i in range(65, 91)] + [str(i) for i in range(10)]

from google.colab import files
uploaded = files.upload()

img_path = list(uploaded.keys())[0]

# Load image
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (64,64))
img_norm = img_resized.astype("float32") / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# Predict
prediction = model.predict(img_input)
pred_idx = np.argmax(prediction)
pred_label = class_labels[pred_idx]

# Show result
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title(f"Predicted: {pred_label}")
plt.show()

print("Predicted Sign:", pred_label)
