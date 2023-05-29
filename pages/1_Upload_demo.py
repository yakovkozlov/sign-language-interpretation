import streamlit as st
import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append('../')
from model_cnn02 import ModelCNN02

st.sidebar.success("Select another demo")

st.title("Do you have a sign image to covert?!")

# creating interface to upload data
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image file :sunglasses:", type=['png', 'jpg'])
#
if uploaded_file is not None:
    data = uploaded_file
    st.write("filename:", data.name)
    st.success('An image has been successfully uploaded! You are doing great!')

    #convert image to test file for prediction - preproc
    uploaded_image = Image.open(data)

    res = cv2.resize(np.array(uploaded_image), dsize=(56, 56), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

    # Reshape to (56, 56, 1)
    reshaped_image = np.reshape(grayscale_image, (56, 56, 1))

    # Save the resized image as a temporary file
    # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #     temp_path = temp_file.name
    #     cv2.imwrite(temp_path, uploaded_image)

    #using the model to predict
    model=ModelCNN02()
    answer = model.predict(reshaped_image)

    st.write("the sign means", answer)
    
    c1, c2, c3 = st.beta_columns(3)
    # Space out the maps so the first one is 2x the size of the other three
    c1, c2, c3 = st.beta_columns((1, 2, 3))

    c1.header("Original")
    c1.image(uploaded_image, use_column_width=True)

    c2.header("Resize")
    c2.image(res, use_column_width=True)
    
    c3.header("Grayscale")
    c3.image(grayscale_image, use_column_width=True)
