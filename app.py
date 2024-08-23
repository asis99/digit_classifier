import streamlit as st
# from model.makeinference import MakePredictions
import sys
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf

def ReadImage(digitImage):
    # Assuming digitImage is a file-like object (e.g., from st.file_uploader)
    # if not isinstance(digitImage, str):
    file_bytes = np.asarray(bytearray(digitImage.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # else:
    image = cv2.imread(digitImage, cv2.IMREAD_GRAYSCALE)

    return image

def MakePredictions(digitImage):
    model_name="./model/digits_recognition_cnn.h5"
    loaded_model = tf.keras.models.load_model(model_name)
    # file_bytes = np.asarray(bytearray(digitImage.read()), dtype=np.uint8)
    # # Step 1: Load the image in grayscale mode
    # image = cv2.imread(file_bytes, cv2.IMREAD_GRAYSCALE)
    image =ReadImage(digitImage=digitImage)
    # Step 2: Invert the image (bitwise not operation)
    image = cv2.bitwise_not(image)

    # Step 3: Define a kernel for dilation (e.g., a 10x10 square kernel)
    kernel = np.ones((10, 10), np.uint8)

    # Step 4: Apply dilation
    dilation = cv2.dilate(image, kernel, iterations=1)

    # Step 5: Resize the image to 28x28
    image = cv2.resize(dilation, (28, 28)) / 255.0

    # Step 6: Add a channel dimension to the image
    img = np.expand_dims(image, axis=-1)

    single_image = np.expand_dims(img, axis=0)

    # Now predict using the loaded model
    predictions_one_hot = loaded_model.predict(single_image)
    predictions = np.argmax(predictions_one_hot, axis=1)
    return {"modelOutput":f"The detected number in the image is {pd.DataFrame(predictions)[0][0]}",
            "Probabilities":predictions_one_hot}







sideBar=st.sidebar # creating a side bar component
inputImage=sideBar.file_uploader(label="Upload a image of your digit", type=["jpg", "png"])
st.logo(image="chip-ai-svgrepo-com.svg")

if inputImage is not None:
        model_name="./model/digits_recognition_cnn.h5"
        loaded_model = tf.keras.models.load_model(model_name)
        # Open the image using PIL
        inputImage = "./model/3.png"
        
        col1, col2 = st.columns(2, gap="large")
        
        file_bytes = np.asarray(bytearray(inputImage), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        # else:
        image = cv2.imread(inputImage, cv2.IMREAD_GRAYSCALE)
        image = cv2.bitwise_not(image)

    # Step 3: Define a kernel for dilation (e.g., a 10x10 square kernel)
        kernel = np.ones((10, 10), np.uint8)

        # Step 4: Apply dilation
        dilation = cv2.dilate(image, kernel, iterations=1)

        # Step 5: Resize the image to 28x28
        image = cv2.resize(dilation, (28, 28)) / 255.0

        # Step 6: Add a channel dimension to the image
        img = np.expand_dims(image, axis=-1)

        single_image = np.expand_dims(img, axis=0)

        # Now predict using the loaded model
        predictions_one_hot = loaded_model.predict(single_image)
        predictions = np.argmax(predictions_one_hot, axis=1)
        st.write(predictions)





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
