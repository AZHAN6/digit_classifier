import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("image_classifier_model.keras")

st.title("Handwritten Digit Classifier")

uploaded_file = st.file_uploader(
    "Upload a 28x28 image of a digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", width=200)

    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to array
    image_array = np.array(image).astype("float32") / 255

    # Flatten
    image_array = image_array.reshape(1, 784)

    # Predict
    prediction = model.predict(image_array)

    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")