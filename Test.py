from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pickle

# Define the filenames
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
image_file = 'out.png'  # Replace with your image file path

# Load the model and label encoder
model = load_model(MODEL_FILENAME)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load and preprocess the image
image = cv2.imread(image_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (20, 20))  # Resize the image to 20x20 pixels
image = np.expand_dims(image, axis=2)
image = np.expand_dims(image, axis=0)

# Make a prediction for the single letter
prediction = model.predict(image)
predicted_letter = lb.inverse_transform(prediction)[0]

print("Predicted letter is:", predicted_letter)
