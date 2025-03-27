import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
from dataset import BaseGreenEarthNetDataset, SeqGreenEarthNetDataset, custom_collate_fn, ADDITIONAL_INFO_DICT, CHANNEL_DICT

# Set page configuration
st.set_page_config(layout="wide", page_title="Model Visualization")

#st.title("Model Visualization")

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

# Create a row for the 5 small images at the top
st.markdown("### Input Images")
top_cols = st.columns(5)


info_list = list(ADDITIONAL_INFO_DICT.keys())

# Load dataloader
ds_train = SeqGreenEarthNetDataset(
    folder="example_preprocessed_dataset/",
    input_channels=["red", "green", "blue"],
    target_channels=["ndvi", "class"],
    additional_info_list=info_list,
    time=True,
    use_mask=True,
)

batch = ds_train[0]


# Create placeholder images for inputs with different colors
input_colors = ["#FFE4E1", "#E6E6FA", "#F0FFF0", "#FFF0F5", "#F5F5DC"]  # Pastel colors
# Add placeholder images to the top row
for i, col in enumerate(top_cols):
    with col:
        img = create_placeholder_image(150, 150, f"Input {i+1}", input_colors[i])
        img = batch["inputs"][i]
        st.image(img, use_container_width=True, caption=f"Input {i+1}")

# Add a slider ranging from 0 to 10 in the input section
st.markdown("### Parameter Control")
slider_value = st.slider("Select a value", min_value=0, max_value=10, value=5, step=1)
st.write(f"Current slider value: {slider_value}")

# Create a row for the two large images below
st.markdown("### Results")
bottom_cols = st.columns(2)

# Add placeholder for model prediction (left)
with bottom_cols[0]:
    pred_img = create_placeholder_image(400, 400, "Model Prediction\n(Not yet implemented)", "lightblue")
    st.image(pred_img, use_container_width=True, caption="Model Prediction")

# Add placeholder for ground truth (right)
with bottom_cols[1]:
    gt_img = create_placeholder_image(400, 400, "Ground Truth", "lightgreen")
    st.image(gt_img, use_container_width=True, caption="Ground Truth")

