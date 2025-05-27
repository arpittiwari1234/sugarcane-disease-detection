import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32

# Load your trained model (ensure it's in the same directory or provide full path)
model = tf.keras.models.load_model('your_model.h5')

# Function to make prediction from uploaded image
def model_prediction(uploaded_image):
    img = Image.open(uploaded_image).resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension

    predictions = model.predict(img_array)
    confidence = round(100 * np.max(predictions[0]), 2)
    predicted_class = np.argmax(predictions[0])
    return predicted_class, confidence

# Add background from local image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpg;base64,{encoded}");
             background-size: cover;
             background-attachment: fixed;
             color: white;
         }}
         .css-1d391kg {{
             background-color: rgba(0, 0, 0, 0.6);
         }}
         .sidebar .sidebar-content {{
             background-color: #004d4d;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

# Add background
add_bg_from_local('images.jpg')

# Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About', 'Disease Recognition'])

# Home Page
if app_mode == 'Home':
    st.header('PLANT DISEASE RECOGNITION SYSTEM')
    st.image('images_of_sugarcan.jpg', use_column_width=True)
    st.markdown('''
        ## Welcome to the Plant Disease Recognition System  
        Upload an image of your plant leaf and get a prediction!
    ''')

# About Page
elif app_mode == "About":
    st.header('About')
    st.markdown('''
        ### About Dataset  
        1. Train (70,295 images)  
        2. Validation (17,572 images)  
        3. Test (33 images)
    ''')

# Prediction Page
elif app_mode == 'Disease Recognition':
    st.header('Disease Recognition')
    uploaded_file = st.file_uploader('Browse your image:', type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        if st.button('Show Image'):
            st.image(uploaded_file, use_column_width=True)
        
        if st.button('Predict'):
            st.write('Our Prediction:')
            predicted_class, confidence = model_prediction(uploaded_file)
            st.success(f'Predicted Class: {predicted_class} with {confidence}% confidence')
