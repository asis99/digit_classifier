import streamlit as st
from model.makeinference import MakePredictions
import sys
import pandas as pd


# sideBar=st.sidebar # creating a side bar component
# inputImage=sideBar.file_uploader(label="Upload a image of your digit", type=["jpg", "png"])
# st.logo(image="chip-ai-svgrepo-com.svg")

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd

# Load the model
model_name = "./model/digits_recognition_cnn.h5"
loaded_model = tf.keras.models.load_model(model_name)

def ReadImage(file):
    """
    Reads an image from a file-like object and converts it to grayscale.
    
    :param file: File-like object from st.file_uploader
    :return: Grayscale image as a numpy array
    """
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None:
            st.error("Failed to decode the image. Please upload a valid image.")
            return None
        return image
    else:
        st.error("No file uploaded.")
        return None

def MakePredictions(image):
    """
    Processes the image and makes predictions using the loaded model.
    
    :param image: Grayscale image as a numpy array
    :return: Dictionary with prediction and probabilities
    """
    if image is None:
        return {"modelOutput": "No image provided", "Probabilities": None}
    
    # Invert the image
    image = cv2.bitwise_not(image)

    # Define a kernel for dilation
    kernel = np.ones((10, 10), np.uint8)

    # Apply dilation
    dilation = cv2.dilate(image, kernel, iterations=1)

    # Resize the image to 28x28
    image = cv2.resize(dilation, (28, 28)) / 255.0

    # Add channel and batch dimensions
    img = np.expand_dims(image, axis=-1)
    single_image = np.expand_dims(img, axis=0)

    # Predict using the loaded model
    predictions_one_hot = loaded_model.predict(single_image)
    predictions = np.argmax(predictions_one_hot, axis=1)

    return {
        "modelOutput": f"The detected number in the image is {predictions[0]}",
        "Probabilities": predictions_one_hot
    }

# Streamlit app layout
st.title("Digit Recognition with Streamlit")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Read and process the image
    image = ReadImage(uploaded_file)
    
    if image is not None:
        # Show the uploaded image
        st.image(image, channels="GRAY", use_column_width=True)
        
        # Make predictions
        results = MakePredictions(image)
        
        # Display results
        st.write(results["modelOutput"])
        if results["Probabilities"] is not None:
            st.write("Probabilities:", results["Probabilities"])




    # try:
#     if inputImage is not None:
#         # Open the image using PIL
#         inputImage = inputImage
        
#         col1, col2 = st.columns([1, 2], gap="large")
        
#         # with col1:
#         # st.image(inputImage, caption="Uploaded Image", use_column_width=True)
#         modelOutput = MakePredictions(digitImage=inputImage)
        
#         # with col2:
#         probData=[i for i in modelOutput.get("Probabilities").tolist()[0]]
#         data=pd.DataFrame({"x":probData})
#         st.write(data)
#         # st.bar_chart(x=[1,2,3,4,5,6,7])
#             # sns.barplot(x=[i for i in range(10)], y=probData)

#         st.write("Model Output:")
#         st.write(modelOutput.get("modelOutput"))
    
#     else:
#         st.markdown(
#         """
#         <h1 style='color: #ff6347;'>AI Powered Digit Recognizer</h1>
#         """, unsafe_allow_html=True
#         )
#         st.divider()
#         col1, col2 = st.columns(2, gap="large")
#         with col1:
#             st.image(image="artificial-intelligence.jpg", width=200)
#         with col2:
#             st.subheader("Instruction for image upoading")
#             st.divider()
#             st.write('''
#                     Please upload an image that has been cropped to a 1:1 ratio, ensuring the digit is centered within the frame & background must be white.
#                     Make sure the digit is written on white paper using any pen or pencil. 
#                     Note that blurred images may not be accurately recognized by the system. 
#                     The model may occasionally misinterpret certain digits, so if the prediction is incorrect,
#                     try to re-crop it, zoom it a little and upload the image. For optimal results, use a high-definition camera to provide clearer 
#                     images, which will help the system recognize the digit more accurately.
#                     ''')
# except Exception as e:
#     print(f"Exception is {str(e)}")
#     line_number = sys.exc_info()[-1].tb_lineno
#     print(f"Error is at line no {line_number}")
