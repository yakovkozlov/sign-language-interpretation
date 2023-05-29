import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

import sys
sys.path.append('../')
from model_cnn02 import ModelCNN02

st.set_page_config(
    page_title="Upload file",
    page_icon=":muscle:"
    )

st.sidebar.success("Select another demo")

st.title("Do you have a sign image to covert?!")

# creating interface to upload data
st.set_option('deprecation.showfileUploaderEncoding', False)

#upload a model to run
with st.beta_expander(":red[Load the pre-trained model] :nerd_face:", expanded=False):
    uploaded_model = st.file_uploader("1. Choose an tensorflow model: ", type=['h5', 'keras'])

    if uploaded_model is not None:
        model_path = "./model.h5"  # Define a path to save the uploaded file
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getvalue())
            model = load_model(model_path)

        st.success('You model has been successfully uploaded! You are doing great!')

#upload an image to test
with st.beta_expander(":blue[Load the image]", expanded=False):
    uploaded_file = st.file_uploader("2. Choose an image file :sunglasses:", type=['png', 'jpg'])

    if uploaded_file is not None:
        data = uploaded_file
        st.write("filename:", data.name)

        #convert image to test file for prediction - preproc
        uploaded_image = Image.open(data)

        # Resize to 100 x 100
        res = cv2.resize(np.array(uploaded_image), dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

        st.write("shape coming in:", res.shape)
        st.write("Expected model shape:", model.input_shape)

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

        # Reshape to (100, 100, 3)
        reshaped_image = np.reshape(res, (100, 100, 3))

        #using the model to predict
    #     model=ModelCNN02()
        answer = model.predict(reshaped_image)

        st.write("the sign means", answer)

        with st.beta_expander(":green[Visualize the preprocessing] :nerd_face:", expanded=False):
            c1, c2, c3 = st.beta_columns(3)
            # Space out the maps so the first one is 2x the size of the other three
            c1, c2, c3, c4 = st.beta_columns((4, 3, 2, 1))

        c1.header("Orig")
        c1.image(uploaded_image, use_column_width=True)

        c2.header("Resize")
        c2.image(res, use_column_width=True)
        
        c3.header("Reshape")
        c3.image(reshaped_image, use_column_width=True)

        c4.header("Not using Grayscale, but we could;)")
        c4.image(grayscale_image, use_column_width=True)
