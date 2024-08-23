import streamlit as st
import pandas as pd
# import seaborn as sns
from model.makeinference import MakePredictions
# import seaborn as sns


sideBar=st.sidebar # creating a side bar component
inputImage=sideBar.file_uploader(label="Upload a any image of Cat or Dog", type=["jpg", "png"])
st.logo(image="chip-ai-svgrepo-com.svg")

if inputImage is not None:
    # Open the image using PIL
    inputImage = inputImage
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.image(inputImage, caption="Uploaded Image", use_column_width=True)
        modelOutput = MakePredictions(digitImage=inputImage)
    
    with col2:
        probData=[i for i in modelOutput.get("Probabilities").tolist()[0]]
        st.bar_chart(data=probData, stack=True, x_label="Predictions", y_label="Probability")

    st.write("Model Output:")
    st.write(modelOutput.get("modelOutput"))
   
else:
    st.markdown(
    """
    <h1 style='color: #ff6347;'>AI Powered Digit Recognizer</h1>
    """, unsafe_allow_html=True
     )
    st.divider()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(image="artificial-intelligence.jpg", width=200)
    with col2:
        st.subheader("Instruction for image upoading")
        st.divider()
        st.write('''
                 Please upload an image that has been cropped to a 1:1 ratio, ensuring the digit is centered within the frame.
                  Make sure the digit is written on white paper using any pen or pencil. 
                 Note that blurred images may not be accurately recognized by the system. 
                 The model may occasionally misinterpret certain digits, so if the prediction is incorrect,
                  try re-crop it, zoom it a little and uploading the image. For optimal results, use a high-definition camera to provide clearer 
                 images, which will help the system recognize the digit more accurately.
                  ''')