import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

# Set page configuration with minimal padding and spacing
st.set_page_config(
    layout="wide", 
    page_title="Model Visualization",
    initial_sidebar_state="collapsed"
)

# Custom CSS to reduce padding and make layout more compact
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stImage {
        margin-bottom: 0.5rem;
    }
    .row-widget.stMarkdown {
        margin-bottom: 0.2rem;
    }
    h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    p {
        margin-bottom: 0.3rem;
    }
    .stMarkdown p {
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a placeholder image
def create_placeholder_image(width, height, text, color="lightgray"):
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img

# Create a compact layout
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("### Inputs")
with col2:
    top_cols = st.columns(5)
    
    # Create placeholder images for inputs with different colors - smaller size
    input_colors = ["#FFE4E1", "#E6E6FA", "#F0FFF0", "#FFF0F5", "#F5F5DC"]
    for i, col in enumerate(top_cols):
        with col:
            img = create_placeholder_image(100, 100, f"Input {i+1}", input_colors[i])
            st.image(img, use_container_width=True, caption=None)
            st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>Input {i+1}</p>", unsafe_allow_html=True)

# Create a row for the two large images below
st.markdown("<h3 style='margin-top: 0.5rem;'>Results</h3>", unsafe_allow_html=True)
bottom_cols = st.columns(2)

# Add placeholder for model prediction (left) - reduced height
with bottom_cols[0]:
    pred_img = create_placeholder_image(350, 250, "Model Prediction\n(Not yet implemented)", "lightblue")
    st.image(pred_img, use_container_width=True, caption=None)
    st.markdown("<p style='text-align: center;'>Model Prediction</p>", unsafe_allow_html=True)

# Add placeholder for ground truth (right) - reduced height
with bottom_cols[1]:
    gt_img = create_placeholder_image(350, 250, "Ground Truth", "lightgreen")
    st.image(gt_img, use_container_width=True, caption=None)
    st.markdown("<p style='text-align: center;'>Ground Truth</p>", unsafe_allow_html=True)

