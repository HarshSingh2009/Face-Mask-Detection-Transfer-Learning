from PIL import Image
import tensorflow as tf
import streamlit as st
from pipeline import PredictionPipeline

st.title('Face Mask Detection')
st.write('This Project is built using CNN (Convolutional Neural Networks) and help in predicti if the person has a mask or not')

st.write('')
st.write('')


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Process the uploaded image here\
    with st.container():
        col1, col2 = st.columns([3, 2])
        col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        # leave some extra space at the top
        col2.text('')
        col2.text('')

        if st.button('Predict!!'):
            pipeline = PredictionPipeline()
            y_pred, y_probs = pipeline.predict(input_img=uploaded_file)
            if y_pred[0][0] == 1:
                col2.snow()
                col2.subheader('Without Face Mask Detected!!')
                acc = '{:.2f}'.format(100*(y_probs[0][0]))
                col2.success(f'Accuracy: {acc}%')
            elif y_pred[0][0] == 0:
                col2.balloons()
                col2.subheader('Face Mask Detected!!')
                acc = '{:.2f}'.format(100*(1-y_probs[0][0]))
                col2.success(f'Accuracy: {acc}%')
            else:
                col2.error('Error!! Model needs shape (224, 224, 3), but your image is of shape (224, 224,4)')