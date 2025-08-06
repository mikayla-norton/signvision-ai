import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Adversarial - SignVision AI',
    page_icon='⚔️',
    layout='wide'
)
st.title("Adversarial Attacks & Robustness")

st.header("Fast Gradient Sign Method (FGSM)")
st.write(r'''FGSM is a white-box attack that generates adversarial examples by using the gradients of the model's loss function with respect to the input image. It perturbs the input image in the direction of the gradient sign, creating an adversarial example that can mislead the model into making incorrect predictions.

The perturbation is controlled by a hyperparameter $\epsilon$, which determines the magnitude of the perturbation. A larger $\epsilon$ results in a more significant perturbation, while a smaller $\epsilon$ creates a subtler change to the input image.
The goal of FGSM is to maximize the loss function $J$ for a given input image $x$ and its true label $y$. The perturbation is computed as follows:
$$
\epsilon = \text{max}_{\delta} \bigl( J(x + \delta, y) - J(x, y) \bigr)
$$
where $\delta$ is the perturbation applied to the input image. The perturbation is then added to the original image to create the adversarial example $x'$:

$$ 
x' = x + \epsilon \,\text{sign}\bigl(\nabla_x J(x,y)\bigr).
$$
''')


st.image("streamlit_app/assets/result_images/whitebox.png", caption="White-box Perturbations", use_column_width=True)


st.subheader("Attack Implementation & Results")
st.markdown("""
- **Attack Method:** FGSM using Foolbox.
- **Total Attacked Images:** 4,350 (87 batches of 50).
- **Average Attack Success (Full Model):** 92.76%.
- **Average Attack Success (Reduced Model):** 98.81%.
""")

st.write("These high misclassification rates highlight the need for adversarial defenses.")
st.image("streamlit_app/assets/result_images/advAttack.png", caption="Sample Adversarial Examples")


