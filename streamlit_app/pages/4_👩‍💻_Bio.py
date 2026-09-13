import streamlit as st

from config import ASSETS_DIR

st.set_page_config(page_title='About Dev - SignVision AI',
                    page_icon='👩‍💻',
                    layout="wide")
st.title("About the Developer - SignVision AI")

col1, col2 = st.columns(2)

col1.subheader("Mikayla Norton")

col1.markdown("###### _noun_")
col1.text("SENIOR DATA SCIENTIST AT HUMANA")
col1.write("")

col1.markdown("###### _verb_")
col1.text("1. INNOVATE AND DESIGN")
col1.text("2. INFLUENCE AND ENERGIZE")
col1.write("")


col1.markdown("###### _adjective_")
col1.text("1. PASSIONATE")
col1.text("2. PRECISE")
col1.text("3. GOAL-ORIENTED")
col1.write("")

col1.markdown('<h4 style="text-align: left;">Who am I?</h4>', unsafe_allow_html=True)
col1.markdown("<p style='text-align: justify;'>Nice to 'e-meet' you! My name is Mikayla, I am the developer of SignVision AI. I am currently a Senior Data Scientist at Humana. I am a passion-centric learner, with a drive to always challenge myself for something new. I hope you've enjoyed my application so far.</p>", unsafe_allow_html=True)
col1.write("")


col2.image(str(ASSETS_DIR / "headshot.jpg"))
