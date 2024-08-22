import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sn
import numpy as np
import pandas as pd
import cv2
# import matplotlib.pyplot as plt
def ReadImage(digitImage):
    # Assuming digitImage is a file-like object (e.g., from st.file_uploader)
    if not isinstance(digitImage, str):
        file_bytes = np.asarray(bytearray(digitImage.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
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

# f=MakePredictions(digitImage='E:/datasets/my_digit_data/x.jpg')
# print(f)