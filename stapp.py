import streamlit as st
import numpy as np
import torch
from PIL import Image
import glob
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model configurations
def build_model(modelname, device):
    configs = {
        "sam2.1-hiera-tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1-hiera-large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    checkpoints = {
        "sam2.1-hiera-tiny": "checkpoints/sam2.1_hiera_tiny.pt",
        "sam2.1-hiera-small": "checkpoints/sam2.1_hiera_small.pt",
        "sam2.1-hiera-base-plus": "checkpoints/sam2.1_hiera_base_plus.pt",
        "sam2.1-hiera-large": "checkpoints/sam2.1_hiera_large.pt",
    }

    cfg, ckpt = configs[modelname], checkpoints[modelname]
    predictor = SAM2ImagePredictor(build_sam2(cfg, ckpt, device))
    return predictor

# List available directories dynamically
def list_folders(base_path=os.path.expanduser("~"), max_depth=None):
    try:
        folders = []
        for root, dirs, _ in os.walk(base_path):
            level = root.replace(base_path, '').count(os.sep)
            if max_depth is None or level < max_depth:
                folders.extend([os.path.join(root, d) for d in dirs])
        return sorted(folders)
    except FileNotFoundError:
        return []

# Load images from directory
def load_images(directory):
    files = []
    for ext in ["*.jpeg", "*.jpg", "*.png"]:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files

# Main Streamlit UI
def main():
    st.title("Segmentation using SAM2")

    available_folders = list_folders()
    directory = st.selectbox("Select Image Directory:", available_folders) if available_folders else None
    modelname = st.selectbox("Select Model:", [
        "sam2.1-hiera-tiny", "sam2.1-hiera-small", "sam2.1-hiera-base-plus", "sam2.1-hiera-large"
    ])

    if st.button("Load Images") and directory:
        full_directory = directory
        st.session_state["image_files"] = load_images(full_directory)
        st.session_state["current_index"] = 0
        st.session_state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        st.session_state["predictor"] = build_model(modelname, st.session_state["device"])
        st.session_state["masks"] = [None] * len(st.session_state["image_files"])
        st.success("Images Loaded Successfully!")

    if "image_files" in st.session_state and st.session_state["image_files"]:
        idx = st.slider("Select Image", 0, len(st.session_state["image_files"])-1, st.session_state["current_index"])
        st.session_state["current_index"] = idx

        image_path = st.session_state["image_files"][idx]
        image = Image.open(image_path).convert("RGB")
        st.image(image, caption=f"Image {idx+1}/{len(st.session_state['image_files'])}", use_column_width=True)

        if st.button("Segment Image"):
            predictor = st.session_state["predictor"]
            predictor.set_image(np.array(image).astype(np.uint8))

            masks, scores, _ = predictor.predict(multimask_output=True)
            best_mask = masks[np.argmax(scores)]

            st.session_state["masks"][idx] = Image.fromarray((best_mask * 255).astype(np.uint8))
            st.image(st.session_state["masks"][idx], caption="Segmented Mask", use_column_width=True)
            st.success("Segmentation Completed!")

        if st.button("Save Mask"):
            save_dir = os.path.join(os.path.expanduser("~"), directory, "masks")
            os.makedirs(save_dir, exist_ok=True)
            mask_path = os.path.join(save_dir, f"mask_{idx}.png")
            st.session_state["masks"][idx].save(mask_path)
            st.success(f"Mask saved at {mask_path}")

if __name__ == "__main__":
    main()

