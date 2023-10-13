from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from imutils import paths
import pickle

progress_folder = ''
LETTER_IMAGES_FOLDER = "AI work\captcha_output"
MODEL_LABELS_FILENAME = "model_labels.dat"
MODEL_FILENAME = "captcha_model.hdf5"

# Load image paths and labels
data = []
labels = []

print("Loading and processing images...")

for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (20, 20))  # Resize the image to 20x20 pixels
    image = np.expand_dims(image, axis=2)
    label = image_file.split(os.path.sep)[-2]

    if label != progress_folder:
        print('Processing Folder', label)
        progress_folder = label

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32") / 255.0

# Binarize labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)
print("done")

# Split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.20, random_state=42)

# Create and compile the model
model = keras.Sequential()
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(34, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# Train the model
print("Training the model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10, verbose=1)
print("Training complete.")

# Save the model
model.save(MODEL_FILENAME)
print(f"Model saved as {MODEL_FILENAME}")

