import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

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
st.header("Bridging Worlds: Real-Time ASL Translation with Robust, Adversarially-Hardened Models")

# → Logos bar
cols = st.columns(11)
logos = [
    'python.png','keras.png','tf.png','opencv.png','streamlit.png',
    'sklearn.png','tqdm.png','pandas.png','seaborn.png','numpy.png','matplotlib.png'
]
for c, logo in zip(cols, logos):
    c.image(Image.open(f'streamlit_app/assets/image-logos/{logo}'), width=40)


st.subheader("Empowering Communication")
st.write("""
SignVision AI advances communication for the deaf and hard-of-hearing by translating ASL alphabet gestures into text with high accuracy, enabling seamless interaction.
""")

st.subheader("Adversarially-Hardened")
st.write("""
To ensure reliability, our system defends against adversarial attacks—subtle perturbations designed to mislead AI models—through rigorous testing and adversarial training.
""")

st.subheader("Objectives")
st.markdown("""
- **Baseline Model Training:** Train a deep neural network on 29 ASL classes (A–Z, space, delete, blank) using 87,000 images.
- **Adversarial Attacks:** Evaluate model vulnerability via the Fast Gradient Sign Method (FGSM).
- **Adversarial Training:** Retrain with adversarial samples to enhance robustness.
- **Robustness Evaluation:** Compare natural vs. adversarial accuracy trade-offs.
""")

st.subheader("Dataset Overview")
st.markdown("""
- 29 distinct classes, including letters A–Z, space, delete, and blank.
- **87,000** labeled images (3,000 per class) collected against a uniform background.
""")

st.header("Success Criteria")
st.markdown("""
- **Baseline Test Accuracy:** ≥93.6%
- **Adversarial Robustness:** Achieve >95% accuracy on perturbed inputs post-training.
""")
