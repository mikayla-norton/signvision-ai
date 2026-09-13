import sys

import numpy as np
import streamlit as st
from PIL import Image

from config import CLASS_NAMES, PROJECT_ROOT, RESULT_IMAGE_DIR

# The plotting helpers live in the project's src/ package; make them importable
# regardless of the directory Streamlit is launched from.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import viz  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(
    page_title='Results - SignVision AI',
    page_icon='📊',
    layout='wide'
)

st.title("Training & Evaluation Results - SignVision AI")


col1, col2 = st.columns([6, 1])

with col1:

    st.header("Data Distribution")
    with open(RESULTS_DIR / "distPlot.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=400)

    st.header("Confusion Matrix")
    cm = np.loadtxt(RESULTS_DIR / "confusion_matrix.csv", delimiter=",")
    fig = viz.plot_confusion_matrix(
        cm,
        classes=CLASS_NAMES,
        normalize=False,
        title_fontsize=12,
        tick_fontsize=8,
        text_fontsize=4,
        figsize=(6, 6),
    )
    st.pyplot(fig)

    # Performance summary metrics
    st.subheader("Model Performance Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Test Accuracy", "93.6%")
    c2.metric("Robust Natural Accuracy", "84%")
    c3.metric("Robust Adversarial Accuracy", "97%")

    st.markdown("""
    The adversarially trained model shows a trade-off: a modest decrease in natural
    accuracy in exchange for substantial robustness against adversarial examples.
    """)
    st.caption(
        "Baseline accuracy is from the training logs (`results/logs.npy`); the "
        "robustness figures are from the adversarial-training experiments in "
        "the CMSE 890 project."
    )

with col2:
    st.header("Model Structure")
    st.image(
        Image.open(RESULT_IMAGE_DIR / "model.png"),
        caption="Model Architecture",
        use_column_width=True,
    )
