import streamlit as st
import pandas as pd 
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

st.title('Brain Tumor Detection')

# Load the saved model outside the prediction function
loaded_model = load_model('cnn.h5')

def prediction(file):
    img = tf.keras.utils.load_img(file, target_size=(512, 512))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Predict the class probabilities
    classes = loaded_model.predict(x)

    # Get the predicted class label
    classes = np.ravel(classes) # convert to 1D array
    idx = np.argmax(classes)
    clas = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor'][idx]

    return clas

uploaded_file = st.file_uploader("Choose a Brain MRI file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
  
if st.button('Predict'): 
    result = prediction(uploaded_file)
    st.write('Prediction is {}'.format(result))