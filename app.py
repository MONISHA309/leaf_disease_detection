import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model (ensure the path is correct)
model = load_model('plant_disease_model.h5')

# Define the class names
class_names = ['Corn common rust', 'Potato Early blight', 'Tomato bacterial spot']

# Function to process the image
def preprocess_image(img):
    # Convert image to RGB (if it's not)
    img = img.convert('RGB')
    
    # Resize image to the input shape expected by the model (256x256)
    img_resized = img.resize((256, 256))
    
    # Convert image to numpy array and normalize
    img_array = np.array(img_resized) / 255.0
    
    # Expand dimensions to match the input shape (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Streamlit UI for file upload
st.title("Leaf Disease Prediction")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(img)

    # Model prediction
    predictions = model.predict(processed_image)
    
    # Display prediction result
    st.write("Predictions: ")
    st.write(predictions)

    # Get the predicted class and display its name
    predicted_class = np.argmax(predictions, axis=1)
    st.write(f"Predicted Class: {class_names[predicted_class[0]]}")
