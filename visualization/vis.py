import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import torch
from dataset import BaseGreenEarthNetDataset, SeqGreenEarthNetDataset, custom_collate_fn, ADDITIONAL_INFO_DICT, CHANNEL_DICT

# Set page configuration
st.set_page_config(layout="wide", page_title="Model Visualization")

# Add custom CSS to change background color
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff;  /* Light blue background */
    }
    </style>
    """, unsafe_allow_html=True)

#st.title("Model Visualization")


# Function to create a placeholder image
def create_placeholder_image(width, height, text, color="lightgray"):
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.set_facecolors(color)
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
    folder="val/",
    input_channels=["red", "green", "blue"],
    target_channels=["evi", "class"],
    additional_info_list=info_list,
    time=True,
    use_mask=True,
)


# take 10 sequences where the overall nan count is less than 5%
filtered = []
to_filter = 100
to_filter = min(len(ds_train), to_filter)
for i in range(to_filter):

    try:
        batch = ds_train[i]
        seq = batch["inputs"]

        total_pixel_count = seq.shape[0] * seq.shape[1] * seq.shape[2] * seq.shape[3]
        nan_count = np.isnan(seq).sum() / total_pixel_count

        target = batch["targets"][0][0]
        # target must be non nan at least 30%
        target_nan_count = np.isnan(target).sum() / (target.shape[0] * target.shape[1])

        if nan_count < 0.3 and target_nan_count < .5:
            filtered.append(i)
    except:
        pass

# write that number of samples were filtered out
st.write(f"Filtered out {len(filtered)} samples out of {to_filter} samples.")


# Add a slider ranging from 0 to 10 in the input section
st.markdown("### Parameter Control")
slider_value = st.slider("Select a sample", min_value=0, max_value=min(10, len(filtered)-1), value=2, step=1)
st.write(f"Displaying sample index: {slider_value}")

# Add model selection dropdown
model_type = st.selectbox(
    "Select Model Type",
    ["Last Image Model", "Simple Model", "Test Model"],
    index=1
)

# Get the batch based on slider value
batch = ds_train[filtered[slider_value]]
gt_img = ds_train[filtered[slider_value]]["targets"]

def preprocess(img):
    # transpose
    img = img.transpose(1, 2, 0)

    # normalize
    mask = ~np.isnan(img)
    min_val = 0.0
    max_val = 0.5


    img = (img - min_val) / (max_val - min_val)

    img = np.nan_to_num(img, nan=1)
    # replace nan with 1

    # clip image
    img = img.clip(0, 1)



    return img


# Create placeholder images for inputs with different colors
# Add placeholder images to the top row
for i, col in enumerate(top_cols):
    with col:
        img = batch["inputs"][i]
        # replace all nan values with 255
        img = preprocess(img)

        # count number of zeros and print
        #st.write(img.shape, img.dtype, np.min(img), np.max(img))
        st.image(img, use_container_width=True, caption=f"Input {i+1}")

# Create a row for the two large images below
st.markdown("### Results")
bottom_cols = st.columns(2)

from simple_model import SimpleConvModel, LastImgModel

# Initialize the model based on dropdown selection
if model_type == "Last Image Model":
    model = LastImgModel()
elif model_type == "Simple Model":  # Simple Model
    model = SimpleConvModel()
    # Load the model state dict
    model.load_state_dict(torch.load("simple_checkpoint.pth")["model_state_dict"])
elif model_type == "Test Model":
    model = SimpleConvModel()
    # Load the model state dict
    model.load_state_dict(torch.load("test_checkpoint.pth")["model_state_dict"])




# PREDICTION
with bottom_cols[0]:
    
    inp = batch["inputs"]
    # replace nan with X
    inp = np.nan_to_num(inp, nan=0.5)

    inp = torch.tensor(inp).double()
    # Convert double to float
    inp = inp.float()
    
    assert inp.shape == (5, 3, 128, 128)
    print(inp.shape, inp.dtype)
    pred_img = model(inp)
    
    # Convert from tensor to numpy for visualization
    pred_img = pred_img.detach().numpy()

    assert pred_img.shape == (1, 128, 128)

    # make it from 1, 128, 128 to 128, 128
    pred_img = pred_img[0]

    assert pred_img.shape == (128, 128)

    # normalize to new min and max
    new_min = -1
    new_max = 1

    # normalize to 0-1
    pred_img = (pred_img - np.min(pred_img)) / (np.max(pred_img) - np.min(pred_img))
    # scale to new min and max
    pred_img = pred_img * (new_max - new_min) + new_min

    # pred_img = batch["targets"][0][0].copy()  # Select one channel if needed  
    # pred_img /= 2

    # # add noise
    # pred_img += np.random.normal(0.7, 0.2, pred_img.shape)
    pred_img = pred_img.clip(-1, 1)

    # invert the image
    pred_img *= -1

    fig, ax = plt.subplots()
    cax = ax.imshow(pred_img  , cmap='viridis')  # You can change the colormap
    fig.colorbar(cax)

    # Display the plot in Streamlit
    st.pyplot(fig)

# Add placeholder for ground truth (right)
with bottom_cols[1]:
    
    gt_img = batch["targets"][0][0]  # Select one channel if needed  
    gt_img = gt_img.clip(-1, 1)

    fig, ax = plt.subplots()
    cax = ax.imshow(gt_img  , cmap='viridis', vmin=-1, vmax=1)  # You can change the colormap
    fig.colorbar(cax)

    # Display the plot in Streamlit
    st.pyplot(fig)



def calculate_rmse(targets, pred):
    from collections import defaultdict
    from sklearn.metrics import mean_squared_error

    class RMSEimagewise():
        def __init__(self, name):
            self.name = name
            self.rmse = defaultdict(lambda: np.array([]))  
            
        def update(self, class_idx, y_gt, y_pred):
            """Update RMSE values for a specific class index."""
            self.rmse[class_idx] = np.append(self.rmse[class_idx], mean_squared_error(y_gt, y_pred))

        def compute(self):
            """Compute the mean RMSE for all class indices."""
            rmse = {"name": self.name}
            for class_idx in self.rmse.keys():
                rmse[class_idx] = float(np.mean(self.rmse[class_idx]))
            return rmse
        
        def __repr__(self):
            """Print the computed RMSE values in a tabular format."""
            rmse = self.compute()
            output = f"{'Class':<10}{'RMSE':<10}\n"
            output += "-" * 20 + "\n"
            for class_idx in rmse.keys():
                if class_idx == "name":
                    continue
                output += f"{class_idx:<10}{rmse[class_idx]:<10.4f}\n"
            return output

        def reset(self):
            """Reset the stored RMSE values."""
            self.rmse = defaultdict(lambda: np.array([]))

        def set_name(self, name):
            """Set a new name for the RMSE tracker."""
            self.name = name

        def get_class_rmse(self, class_idx):
            """Retrieve the RMSE values for a specific class index."""
            return self.rmse[class_idx] if class_idx in self.rmse else np.array([])
    
    stats = RMSEimagewise("RMSE")


    assert pred.shape == (1, 128, 128)
    evi = targets[0:1]
    assert evi.shape == (1, 128, 128)
    class_mask = targets[1:2]
    assert class_mask.shape == (1, 128, 128)

    CLASSES2EVAL = [10, 30, 40] # Only evaluate on these classes


    for class_idx in CLASSES2EVAL:

        # NOTE: The evi channel should be in range [-1, 1], but due to the preprocessing and
        #  noise on cameras, it might not be the case -> we filter out invalid values (consider 
        # to adjust the evi channels in train phase)
        valid_mask = (class_mask == class_idx) & (~np.isnan(evi)) & (evi >= -1) & (evi <= 1)
 
        gt = evi[valid_mask]
        pred_act = pred[valid_mask]
    
        if len(gt) == 0:
            continue

        stats.update(class_idx, gt, pred_act)

    # return for each class
    return {k: float(stats.get_class_rmse(k)[0]) for k in CLASSES2EVAL}

# Calculate rmse
unsqueezed_pred = np.expand_dims(pred_img, axis=0)
RMSE = calculate_rmse(targets=batch["targets"][0], pred=unsqueezed_pred)

st.write(f"RMSE 10 : {RMSE[10]:.5f} (zalesněná oblast)")
st.write(f"RMSE 30 : {RMSE[30]:.5f} (louky a pastviny)")
st.write(f"RMSE 40 : {RMSE[40]:.5f} (pole)")
