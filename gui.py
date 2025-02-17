import sys
import pathlib
import glob
import numpy as np
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sv_ttk


def initialize_model(model_identifier, device_type):
    configurations = {
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

    cfg_path, ckpt_path = configurations[model_identifier], checkpoints[model_identifier]
    return SAM2ImagePredictor(build_sam2(cfg_path, ckpt_path, device_type))


class ImageLoader:
    def __init__(self, img_directory):
        self.image_files = []
        for ext in ["*.jpeg", "*.jpg", "*.png"]:
            self.image_files.extend(glob.glob(f"{img_directory}/{ext}"))
        self.img_directory = img_directory

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        return np.array(image)

    def __len__(self):
        return len(self.image_files)


class ImageSegmentationTool:
    def __init__(self, img_folder, model_id):
        self.device = "cuda"
        self.brush_size = 8
        self.last_segment_color = np.array([0, 0, 1])
        self.segment_colors = []
        self.draw_option = "box"
        self.annotation_mode = "positive"

        self.img_predictor = initialize_model(model_id, self.device)
        self.output_directory = pathlib.Path("./segmented_outputs")
        self.current_index = 0
        self.load_image_data(img_folder)
        self.setup_ui()

    def load_image_data(self, img_folder=None):
        if img_folder is None:
            img_folder = tkinter.filedialog.askdirectory(title="Choose Image Directory")
            if not img_folder:
                return

        self.output_directory = pathlib.Path("./segmented_outputs")
        self.image_dataset = ImageLoader(img_folder)
        self.segmented_masks = [None] * len(self.image_dataset)
        self.annotation_ids = [1] * len(self.image_dataset)
        self.initialize_processing()

    def initialize_processing(self):
        self.current_image = self.image_dataset[self.current_index]
        self.img_predictor.set_image(self.current_image)
        self.annotations = [{"positive": [], "negative": [], "box": None} for _ in range(len(self.image_dataset))]

    def apply_segmentation(self):
        if self.segmented_masks[self.current_index] is None:
            self.segmented_masks[self.current_index] = np.zeros(self.current_image.shape[:2], dtype=np.int32)

        ann_id = self.annotation_ids[self.current_index]
        mask_prompt = {}

        positive_pts = self.annotations[self.current_index]["positive"]
        negative_pts = self.annotations[self.current_index]["negative"]
        bounding_box = self.annotations[self.current_index]["box"]

        points = np.array(positive_pts + negative_pts, dtype=np.float32)
        if len(points) != 0:
            labels = np.array([1] * len(positive_pts) + [0] * len(negative_pts), dtype=np.int32)
            mask_prompt["point_coords"] = points
            mask_prompt["point_labels"] = labels

        if bounding_box is not None:
            mask_prompt["box"] = bounding_box

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, scores, _ = self.img_predictor.predict(**mask_prompt, multimask_output=True)
            chosen_mask = masks[np.argsort(scores)[::-1][0]]

        self.segmented_masks[self.current_index][
            np.logical_and(self.segmented_masks[self.current_index] == 0, chosen_mask == 1)
        ] = ann_id

    def update_canvas(self, event=None):
        self.canvas.delete("all")

        current_image = self.current_image.copy() / 255.0
        overlay_image = current_image.copy()

        if len(self.segment_colors) < self.annotation_ids[self.current_index]:
            self.segment_colors.append(np.random.rand(3))

        for i in range(1, self.annotation_ids[self.current_index]):
            mask = self.segmented_masks[self.current_index] == i
            overlay_image[mask] = 0.5 * current_image[mask] + 0.5 * self.segment_colors[i - 1]

        img = Image.fromarray((overlay_image * 255).astype(np.uint8))
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def save_segmentation_results(self):
        if not self.output_directory.exists():
            self.output_directory.mkdir()
        for src_file, mask in zip(self.image_dataset.image_files, self.segmented_masks):
            if mask is not None:
                mask = Image.fromarray(mask.astype(np.uint8))
                output_file = self.output_directory / (pathlib.Path(src_file).stem + ".png")
                mask.save(output_file)

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Image Segmentation Tool")

        self.canvas = tk.Canvas(self.root, width=512, height=512)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.save_button = ttk.Button(self.root, text="Save Masks", command=self.save_segmentation_results)
        self.save_button.pack(side=tk.TOP, pady=5)

        self.load_button = ttk.Button(self.root, text="Open Directory", command=self.load_image_data)
        self.load_button.pack(side=tk.TOP, pady=5)

        self.quit_button = ttk.Button(self.root, text="Exit", command=self.root.quit)
        self.quit_button.pack(side=tk.TOP)

        sv_ttk.set_theme("light")
        self.root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_directory> <model_name>")
        sys.exit(1)

    img_folder = sys.argv[1]
    model_name = sys.argv[2]
    ImageSegmentationTool(img_folder, model_name)

