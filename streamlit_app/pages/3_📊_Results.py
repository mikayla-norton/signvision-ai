import streamlit as st
import numpy as np
from PIL import Image
import itertools
import viz
from evaluate import evaluate
import pandas as pd

st.set_page_config(
    page_title='Results - SignVision AI',
    page_icon='📊',
    layout='wide'
)

st.title("Training & Evaluation Results - SignVision AI")


col1, col2 = st.columns([6, 1])

with col1:

    st.header("Data Distribution")
    st.components.v1.html(open('distPlot.html','r').read(), height=400)

    st.header("Confusion Matrix")
    cm = np.loadtxt('confusion_matrix.csv', delimiter=',')
    fig = viz.plot_confusion_matrix(
        cm,
        classes=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['delete','nothing','space'],
        normalize=False,
        title_fontsize=12,   # tweak as you like
        tick_fontsize=8,
        text_fontsize=4,
        figsize=(6,6)
    )
    st.pyplot(fig)

    # Performance summary metrics
    st.subheader("Model Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Test Accuracy", "93.6%")
    c2.metric("Robust Natural Accuracy", "84%")
    c3.metric("Robust Adversarial Accuracy", "97%")

    st.markdown("""
    The adversarially trained model shows a trade-off: a slight decrease in natural accuracy in exchange for substantial robustness improvements against adversarial examples.
    """)

with col2:
    st.header("Model Structure")
    st.image(Image.open('assets/result_images/model.png'), caption="Model Architecture", use_column_width=True)