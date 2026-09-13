"""Shared configuration for the SignVision AI Streamlit app.

Keeps the things that were previously duplicated across pages in one place:
class labels, filesystem paths, and the (cached) classifier itself.
"""

from pathlib import Path

import streamlit as st

# --- Paths -----------------------------------------------------------------
# Anchor everything to this file's location so the app works no matter which
# directory Streamlit is launched from.
APP_DIR = Path(__file__).resolve().parent          # .../streamlit_app
PROJECT_ROOT = APP_DIR.parent                      # repo root
ASSETS_DIR = APP_DIR / "assets"
MODEL_DIR = ASSETS_DIR / "models"
RESULT_IMAGE_DIR = ASSETS_DIR / "result_images"
LOGO_DIR = ASSETS_DIR / "image-logos"

CLASSIFIER_WEIGHTS = MODEL_DIR / "ASL_DNN.weights.h5"

# --- Labels ----------------------------------------------------------------
# The single source of truth for class names. The dataset has 29 classes:
# the 26 letters plus three control gestures.
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["delete", "nothing", "space"]

# The trained weights expose 30 output units; the 30th is an unused artifact
# of training (the dataset only has 29 labelled classes). We build the model
# with 30 units so the weights load cleanly, but only ever surface the 29
# real labels. See MODEL_CARD.md for details.
NUM_OUTPUT_UNITS = 30
INPUT_SHAPE = (50, 50, 3)


def label_for(index: int) -> str:
    """Map a predicted class index to a human-readable label.

    Guards against the unused 30th output unit so an out-of-range prediction
    degrades gracefully instead of raising IndexError.
    """
    if 0 <= index < len(CLASS_NAMES):
        return CLASS_NAMES[index]
    return "Unrecognized"


@st.cache_resource(show_spinner="Loading the classifier…")
def load_classifier():
    """Build the VGG16-based classifier and load its trained weights.

    Cached so the network is constructed once per session rather than on every
    Streamlit rerun.
    """
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.models import Model

    base = VGG16(weights=None, include_top=False, input_shape=INPUT_SHAPE)
    x = Flatten()(base.output)
    out = Dense(NUM_OUTPUT_UNITS, activation="softmax", name="dense")(x)
    model = Model(inputs=base.input, outputs=out)

    if not CLASSIFIER_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Model weights not found at {CLASSIFIER_WEIGHTS}. "
            "See README.md for where to obtain them."
        )
    model.load_weights(str(CLASSIFIER_WEIGHTS))
    model.compile()
    return model
