from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
import base64
## tensorflow moodel prediction 


#def model_prediction(image):
#    model = tf.keras.models.load_model("model.keras")
#    class_names = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
#    img = Image.open(image).resize((256,256))
#    img = img.resize((256, 256))  # use your model's image size
#    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
#    #img_array = img_array.reshape(1, -1)
#    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
#    predictions = model.predict(img_array)
    
#    return np.argmax(predictions)

def model_prediction(image):
    model = tf.keras.models.load_model("model.keras")
    class_names = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
    
    img = Image.open(image).resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_index = np.argmax(predictions)
    
    return predicted_index, confidence

# Streamlit file uploader
#test_images = st.file_uploader("Upload leaf image", type=["jpg", "png"])

#if test_images is not None:
    #result_index = model_prediction(test_images)
   # st.success(f"Predicted class index: {result_index}")






#background image
def add_bg_from_local(image_file, opacity=0.5):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(211, 211, 211, {opacity}), rgba(211, 211, 211, {opacity})), 
                        url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
add_bg_from_local('images.jpg', opacity=0.7)

  # <-- Use your own background image file


#sidebar 
st.sidebar.title('Deshboard')
app_mode=st.sidebar.selectbox('Select Page',['Home','About','Disease Recognition'])
#st.sidebar.image("image_of_sugarcan.jpg", use_column_width=True)

#home page 
if(app_mode=='Home'):
    st.markdown("""
    <h1 style='text-align: center; color: Black; font-size: 36px;'>
        🌱 Plant Disease Recognition System
    </h1>
""", unsafe_allow_html=True)

   # st.header('Plant Disease Recognition')
   
    image_path='images_of_sugarcan.jpg'
    with open(image_path, "rb") as f:
        img_data = f.read()
    img_base64 = base64.b64encode(img_data).decode()
    st.markdown(f"""
    <div style="text-align: center; margin-top: 20px;">
        <img src="data:image/jpeg;base64,{img_base64}" 
             style="width: 80%; max-width: 500px; border: 5px solid #4CAF50; 
                    border-radius: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);" />
    </div>
""", unsafe_allow_html=True)
    #st.image(image_path,use_container_width=True)
    st.markdown ('''


            <div style= 'font-size: 20px; margin-top:30px;color:black'>     
                 <p>  
                 Welcome to the plant Disease Recognition System!.
            </div>
            <div style="font-size:15px; color:black;">
                 <h2>Project Aim </h2>     
                <p>
                 Our mission is to help in identifying plant disease efficiently. Upload an image of a lant, and our system 
                 will analyse it to detect any signs of desease. Together, let's protect our  crop and ensure a healthier harvest!. 
                </p>
             </div>                
            <div style = 'color: black;'>
                <h2>How It Works</h2> 
                <p> 
                 <strong>Upload Image:</strong>
                 <span>Go to the Disease Recognition page and upload an image of a plant with suspected diseases.</span>
                 </p>
                 <p>
                 <strong>Analysis:</strong>
                 <span>Our system will process the image using advanced algorithem to identify potemtial diseases.</span>
                 </p>
                 <p>
                 <strong>Results:</strong>
                 <span>View the results and recomendation for further action.</span> 
                  </p>

            </div>
            <div style = 'color:black;'>
                 <h2> Get Started </h2>
                 <p>Click on the <strong>Disease Recognition</strong> page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!.</p>
            </div>
            <div style= " color:black;">
                 <h2>About Us </h2>
                 <p>
                 Learn more about the project, our team, and oour goals on the <strong>About Page</strong> 
                 </p>
                 
            </div>     
''', unsafe_allow_html=True)
# about pages 
elif(app_mode=='About'):
    st.header('About')
    st.markdown("""  
                <div>
                <h2>About Dataset</h2>
                This dataset is consist of about ___ images of healthy and diseased crop leaves is categorized into __ different classes. The total dataset is divided into ____ratio of training and validation set preseving the directory structure. A new directory containing ___ tst images is created later for prediction puspose.  
                
    ### Content
    1. Train(70295 images)
    2. valid (17572 images)
    3. Test (33 images )
                </div>                                 
""", unsafe_allow_html=True)    
# preddiction pages 
elif(app_mode=='Disease Recognition'):
    st.header('Disease Recognition')

    test_images = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if test_images is not None:
        
        if st.button('Show image:'):
            st.image(test_images,use_container_width=True)
    #prdict button 
        '''
        if st.button('Predict'):
            #st.write('Our Prediction') 
            #esult_index=model_prediction(test_images)
            #st.success(f"Predicted class : {result_index}")
            #if st.button('Predict'):
            st.write('Our Prediction')
            result_index = model_prediction(test_images)
            class_names =  ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]  # example

            # Get class name using the index
            predicted_class = class_names[result_index]
        
            st.success(f"Predicted class: {predicted_class}")
        '''
        if st.button('Predict'):
            st.write('Our Prediction')
            
            predicted_index, confidence = model_prediction(test_images)
            class_names = ["Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]
            
            threshold = 0.7  # You can adjust this based on model performance

            if confidence >= threshold:
                predicted_class = class_names[predicted_index]
                st.success(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f})")
            else:
                st.error("Image not matched from dataset (Low confidence)") 
                
