import streamlit as st
from model.makeinference import MakePredictions
import sys
import pandas as pd
import cv2


sideBar=st.sidebar # creating a side bar component
inputImage=sideBar.file_uploader(label="Upload a image of your digit", type=["jpg", "png"])
st.logo(image="chip-ai-svgrepo-com.svg")
try:
    if inputImage is not None:
        st.image("./model/3.png", caption="Uploaded Image", use_column_width=True)
        st.write(cv2.imread("./model/3.png"))
        modelOutput = MakePredictions(digitImage="./model/3.png")
        st.write(modelOutput)
        probData=[i for i in modelOutput.get("Probabilities").tolist()[0]]
        data=pd.DataFrame({"x":probData})
        st.write(data)
except Exception as e:
    st.write(f"{str(e)}")



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
