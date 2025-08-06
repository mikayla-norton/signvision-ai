import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf           # <— use tf.keras
st.title("Live Classification - SignVision AI")
uploaded = st.file_uploader("Upload an image:", type=['png','jpg','jpeg'])
if uploaded:
    with st.spinner("Classifying…"):
        model = tf.keras.models.load_model('streamlit_app/assets/models/ASL_DNN_3D.h5', compile=False)
    # 1) resize to model input
    expected_h, expected_w = model.input_shape[1:3]
    img = Image.open(uploaded).convert('RGB').resize((expected_w, expected_h))

    # 2) layout: two columns, image in the narrow one
    col_img, col_pred = st.columns([1, 2])
    with col_img:
        st.image(img, caption='Input Image', width=200)   # shrink to 150px wide

    # 3) run prediction
    x = np.array(img)[None, ...] / 255.0
    preds = model.predict(x)[0]
    idx   = np.argmax(preds)
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['delete','nothing','space']

    with col_pred:
        st.markdown(f"### Predicted: **{labels[idx]}**")
        st.markdown(f"**Confidence:** {preds[idx]:.1%}")
