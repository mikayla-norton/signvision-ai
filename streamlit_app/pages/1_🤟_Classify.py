import numpy as np
import streamlit as st
from PIL import Image

from config import INPUT_SHAPE, label_for, load_classifier

st.title("Live Classification - SignVision AI")
st.write(
    "Upload a photo of a single ASL alphabet gesture and the model will predict "
    "which letter (or control gesture) it shows."
)

model = load_classifier()

uploaded = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    if min(img.size) < INPUT_SHAPE[0]:
        st.warning(
            f"This image is quite small ({img.size[0]}×{img.size[1]}). "
            "Predictions may be unreliable for images below "
            f"{INPUT_SHAPE[0]}×{INPUT_SHAPE[1]}."
        )

    with st.spinner("Classifying…"):
        h, w = INPUT_SHAPE[:2]
        resized = img.resize((w, h))
        x = np.asarray(resized)[None, ...] / 255.0
        preds = model.predict(x)[0]

    # Rank predictions and keep the 29 real classes.
    order = np.argsort(preds)[::-1]
    top_idx = int(order[0])
    top_conf = float(preds[top_idx])

    col_img, col_pred = st.columns([1, 2])
    with col_img:
        st.image(img, caption="Input image", width=200)

    with col_pred:
        st.markdown(f"### Predicted: **{label_for(top_idx)}**")
        st.markdown(f"**Confidence:** {top_conf:.1%}")

        if top_conf < 0.50:
            st.info(
                "Confidence is low. Try a clearer, well-cropped image of a "
                "single hand against a plain background."
            )

        st.markdown("**Other likely classes**")
        for idx in order[1:4]:
            st.write(f"{label_for(int(idx))} — {preds[int(idx)]:.1%}")

    st.caption(
        "Note: this classifies static images of individual ASL alphabet "
        "gestures. It is not a real-time or continuous ASL translation system."
    )
