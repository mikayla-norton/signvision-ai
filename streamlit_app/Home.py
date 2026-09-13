import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from config import LOGO_DIR

# → Graphing style
plt.rcParams.update({
    'text.color': "white",
    'axes.labelcolor': "white",
    'axes.edgecolor': "white",
    'xtick.color': "white",
    'ytick.color': "white",
    'figure.facecolor': "0F1116",
    'axes.facecolor': "0F1116"
})

# → Page config
st.set_page_config(
    page_title='SignVision AI',
    page_icon='🏠',
    layout='wide'
)

# → Title & tagline
st.title("SignVision AI")
st.header("ASL Alphabet Image Classification and Adversarial Robustness")

# → Logos bar
cols = st.columns(13)
logos = [
    'python.png','keras.png','tf.png', 'foolbox.png', 'eagerpy.png','opencv.png','streamlit.png',
    'sklearn.png','tqdm.png','pandas.png','seaborn.png','numpy.png','matplotlib.png'
]
for c, logo in zip(cols, logos):
    c.image(Image.open(str(LOGO_DIR / logo)), width=40)


st.write("""
SignVision AI is a computer-vision project that classifies static images of
American Sign Language (ASL) alphabet gestures, and then studies how that
classifier behaves when its inputs are deliberately perturbed. I built it as my
graduate project for CMSE 890 at Michigan State University. The question that
motivated it: image classifiers can be fooled by tiny, targeted changes to their
inputs, so how vulnerable is an ASL classifier, and can adversarial training make
it more robust?
""")

st.subheader("What it does")
st.write("""
The model takes a single uploaded image of an ASL alphabet gesture and predicts
its class. This is static-image classification of the ASL alphabet, not
real-time or continuous sign-language translation.
""")

st.subheader("Why robustness matters")
st.write("""
Adversarial examples are inputs with small perturbations that are nearly
invisible to a person but cause a model to misclassify with high confidence. For
any classifier meant to support communication, that failure mode is worth
understanding directly rather than assuming it away. This project measures it
with the Fast Gradient Sign Method (FGSM) and then retrains the model on
adversarial examples to see how much robustness improves.
""")

st.subheader("The dataset")
st.markdown("""
- **29 classes:** the letters A–Z plus three control gestures (`delete`, `nothing`, `space`).
- **~87,000 labeled images** (roughly 3,000 per class) captured against a uniform background.
""")

st.subheader("Key results")
st.markdown("""
- **Baseline test accuracy:** ~93.6% on clean images.
- **Adversarial vulnerability:** the baseline model is easily fooled by FGSM perturbations.
- **After adversarial training:** natural accuracy drops modestly while accuracy on
  adversarial inputs rises substantially. This is the classic robustness trade-off.
""")
st.caption("See the Results and Adversarial pages for the details behind these numbers.")
