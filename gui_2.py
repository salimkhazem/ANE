# coding: utf-8
import sys
import glob
import torch
import sv_ttk
import pathlib
import numpy as np
import tkinter as tk
import tkinter.filedialog

from tkinter import ttk
from PIL import Image, ImageTk
from sam2.build_sam import build_sam2
from collections import OrderedDict
from sam2.sam2_image_predictor import SAM2ImagePredictor


def create_predictor(config_key, compute_device):
    config_files = {
        "sam2.1-hiera-tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1-hiera-small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1-hiera-base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1-hiera-large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    checkpoint_files = {
        "sam2.1-hiera-tiny": "checkpoints/sam2.1_hiera_tiny.pt",
        "sam2.1-hiera-small": "checkpoints/sam2.1_hiera_small.pt",
        "sam2.1-hiera-base-plus": "checkpoints/sam2.1_hiera_base_plus.pt",
        "sam2.1-hiera-large": "checkpoints/sam2.1_hiera_large.pt",
    }

    cfg_file = config_files[config_key]
    ckpt_file = checkpoint_files[config_key]

    segmenter = SAM2ImagePredictor(build_sam2(cfg_file, ckpt_file, compute_device))
    return segmenter


class ImageCollector:
    def __init__(self, folder_path, compute_device):
        self.image_paths = []
        for ext in ["*.jpeg", "*.jpg", "*.png"]:
            self.image_paths.extend(glob.glob(f"{folder_path}/{ext}"))
        self.folder_path = folder_path
        self.compute_device = compute_device

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img_array = np.array(img.convert("RGB"))
        return img_array

    def __len__(self):
        return len(self.image_paths)


class InteractiveSegmentationTool:
    def __init__(self, folder_path, config_key, device):
        self.compute_device = device  # or "cuda" if available
        self.offload_video = False
        self.offload_state = False

        self.marker_size = 10
        self.current_mask_color = np.array([0, 0, 1])
        self.color_palette = []
        self.positive_marker = "red"
        self.negative_marker = "green"
        self.box_active = False
        self.input_mode = "box"  # options: "box" or "point"
        self.point_type = "positive"  # options: "positive" or "negative"

        self.region_top_left = (0, 0)
        self.region_bottom_right = (0, 0)

        self.segmenter = create_predictor(config_key, self.compute_device)

        self.export_dir = pathlib.Path("./masks")
        self.current_index = 0
        self.load_images(folder_path)

        self.initialize_ui()

    def load_images(self, folder_path=None):
        refresh = False
        if folder_path is None:
            refresh = True
            folder_path = tkinter.filedialog.askdirectory(
                initialdir=self.image_collection.folder_path,
                title="Select an image folder",
            )
            if not folder_path:
                return
        self.export_dir = pathlib.Path("./masks")

        self.image_collection = ImageCollector(folder_path, self.compute_device)

        self.mask_storage = [None] * len(self.image_collection)
        self.annotation_counter = [1] * len(self.image_collection)
        self.initialize_segmentation()

        if refresh:
            self.image_slider.configure(to=len(self.image_collection) - 1)
            self.refresh_display()

    def initialize_segmentation(self):
        self.current_img = self.image_collection[self.current_index]
        self.segmenter.set_image(self.image_collection[self.current_index])
        self.user_prompts = []
        for _ in range(len(self.image_collection)):
            self.user_prompts.append({"positive": [], "negative": [], "box": None})
        self.box_start_x = None
        self.box_start_y = None
        self.box_end_x = None
        self.box_end_y = None

    def run_segmentation(self):
        if self.mask_storage[self.current_index] is None:
            self.mask_storage[self.current_index] = np.zeros(
                self.current_img.shape[:2], dtype=np.int32
            )
        prev_mask = (
            self.mask_storage[self.current_index][...]
            == self.annotation_counter[self.current_index]
        )
        self.mask_storage[self.current_index][prev_mask] = 0

        ann_id = self.annotation_counter[self.current_index]
        seg_input = {}

        pos_points = self.user_prompts[self.current_index]["positive"]
        neg_points = self.user_prompts[self.current_index]["negative"]
        bbox = self.user_prompts[self.current_index]["box"]

        if pos_points or neg_points:
            all_points = np.array(pos_points + neg_points, dtype=np.float32)
            labels = np.array(
                ([1] * len(pos_points)) + ([0] * len(neg_points)), dtype=np.int32
            )
            seg_input["point_coords"] = all_points
            seg_input["point_labels"] = labels

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            seg_input["box"] = [x1, y1, x2, y2]

        with torch.inference_mode(), torch.autocast(
            self.compute_device, dtype=torch.bfloat16
        ):
            masks, scores, _ = self.segmenter.predict(
                **seg_input,
                multimask_output=True,
            )
            sorted_idx = np.argsort(scores)[::-1]
            masks = masks[sorted_idx[0]]
        bg_mask = self.mask_storage[self.current_index] == 0
        self.mask_storage[self.current_index][np.logical_and(bg_mask, masks == 1.0)] = (
            ann_id
        )

    def refresh_display(self, event=None):
        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        img_copy = self.current_img.copy()
        normalized_img = img_copy / 255.0

        if len(self.color_palette) < self.annotation_counter[self.current_index]:
            self.color_palette.append(np.random.rand(3))

        overlay = normalized_img.copy()
        for i in range(1, self.annotation_counter[self.current_index]):
            mask_area = self.mask_storage[self.current_index] == i
            overlay[mask_area] = (
                0.5 * normalized_img[mask_area] + 0.5 * self.color_palette[i - 1]
            )

        current_mask = (
            self.mask_storage[self.current_index]
            == self.annotation_counter[self.current_index]
        )
        overlay[current_mask] = (
            0.5 * normalized_img[current_mask] + 0.5 * self.current_mask_color
        )

        display_img = Image.fromarray((overlay * 255).astype(np.uint8))

        canvas_ratio = canvas_w / canvas_h
        img_ratio = display_img.width / display_img.height
        if canvas_ratio > img_ratio:
            new_w = int(canvas_h * img_ratio)
            if new_w == 0:
                new_w = canvas_w
            resized = display_img.resize((new_w, canvas_h))
            pad = (canvas_w - new_w) // 2
            display_img = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            display_img.paste(resized, (pad, 0))
            self.region_top_left = (pad, 0)
            self.region_bottom_right = (pad + new_w, canvas_h)
        else:
            new_h = int(canvas_w / img_ratio)
            if new_h == 0:
                new_h = canvas_h
            resized = display_img.resize((canvas_w, new_h))
            pad = (canvas_h - new_h) // 2
            display_img = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
            display_img.paste(resized, (0, pad))
            self.region_top_left = (0, pad)
            self.region_bottom_right = (canvas_w, pad + new_h)

        self.img_tk = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

        pos_pts = self.user_prompts[self.current_index]["positive"]
        neg_pts = self.user_prompts[self.current_index]["negative"]
        bbox = self.user_prompts[self.current_index]["box"]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1 = self.convert_coords(x1, y1)
            x2, y2 = self.convert_coords(x2, y2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

        for x, y in pos_pts:
            sx, sy = self.convert_coords(x, y)
            self.canvas.create_oval(
                sx - self.marker_size // 2,
                sy - self.marker_size // 2,
                sx + self.marker_size // 2,
                sy + self.marker_size // 2,
                fill=self.positive_marker,
            )
        for x, y in neg_pts:
            sx, sy = self.convert_coords(x, y)
            self.canvas.create_oval(
                sx - self.marker_size // 2,
                sy - self.marker_size // 2,
                sx + self.marker_size // 2,
                sy + self.marker_size // 2,
                fill=self.negative_marker,
            )

    def screen_to_image_coords(self, x, y):
        img_x = (
            (x - self.region_top_left[0])
            / (self.region_bottom_right[0] - self.region_top_left[0])
            * self.current_img.shape[1]
        )
        img_y = (
            (y - self.region_top_left[1])
            / (self.region_bottom_right[1] - self.region_top_left[1])
            * self.current_img.shape[0]
        )
        return img_x, img_y

    def convert_coords(self, x, y):
        screen_x = (
            x
            / self.current_img.shape[1]
            * (self.region_bottom_right[0] - self.region_top_left[0])
            + self.region_top_left[0]
        )
        screen_y = (
            y
            / self.current_img.shape[0]
            * (self.region_bottom_right[1] - self.region_top_left[1])
            + self.region_top_left[1]
        )
        return screen_x, screen_y

    def add_input_point(self, xy):
        scr_x, scr_y = xy
        img_x, img_y = self.screen_to_image_coords(scr_x, scr_y)
        if (
            img_x < 0
            or img_x >= self.current_img.shape[1]
            or img_y < 0
            or img_y >= self.current_img.shape[0]
        ):
            print("Point out of bounds")
            return
        if self.point_type == "positive":
            self.user_prompts[self.current_index]["positive"].append([img_x, img_y])
        else:
            self.user_prompts[self.current_index]["negative"].append([img_x, img_y])

    def handle_mouse_press(self, event):
        start_x, start_y = self.screen_to_image_coords(event.x, event.y)
        if (
            start_x < 0
            or start_x >= self.current_img.shape[1]
            or start_y < 0
            or start_y >= self.current_img.shape[0]
        ):
            print("Point out of bounds")
            return
        if self.input_mode == "box":
            self.box_start_x, self.box_start_y = start_x, start_y
            self.box_active = True
        else:
            self.add_input_point((event.x, event.y))
        self.run_segmentation()
        self.refresh_display()

    def handle_mouse_drag(self, event):
        if self.input_mode == "box" and self.box_active:
            self.box_end_x, self.box_end_y = self.screen_to_image_coords(
                event.x, event.y
            )
            self.user_prompts[self.current_index]["box"] = [
                min(self.box_start_x, self.box_end_x),
                min(self.box_start_y, self.box_end_y),
                max(self.box_start_x, self.box_end_x),
                max(self.box_start_y, self.box_end_y),
            ]
            self.run_segmentation()
            self.refresh_display()

    def handle_mouse_release(self, event):
        if self.box_active:
            self.box_active = False
            if self.input_mode == "box":
                self.run_segmentation()
                self.refresh_display()

    def process_keystroke(self, event):
        if event.char == "p":
            self.input_mode = "point"
            self.point_type = "positive"
            self.mode_label.config(text="Mode:\nPoint (Positive)")
        elif event.char == "n":
            self.input_mode = "point"
            self.point_type = "negative"
            self.mode_label.config(text="Mode:\nPoint (Negative)")
        elif event.char == "b":
            self.input_mode = "box"
            self.mode_label.config(text="Mode:\nBox")
        elif event.keysym == "Left":
            self.switch_image(max(0, self.current_index - 1))
            self.image_slider.set(self.current_index)
        elif event.keysym == "Right":
            self.switch_image(
                min(len(self.image_collection) - 1, self.current_index + 1)
            )
            self.image_slider.set(self.current_index)
        elif event.keysym == "Next":
            self.switch_image(max(0, self.current_index - 10))
            self.image_slider.set(self.current_index)
        elif event.keysym == "Prior":
            self.switch_image(
                min(len(self.image_collection) - 1, self.current_index + 10)
            )
            self.image_slider.set(self.current_index)
        elif event.char == "r":
            self.user_prompts[self.current_index] = {
                "positive": [],
                "negative": [],
                "box": None,
            }
            self.box_start_x = None
            self.box_start_y = None
            self.box_end_x = None
            self.box_end_y = None
            self.mask_storage[self.current_index][...] = 0
            self.annotation_counter[self.current_index] = 1
            self.refresh_display()
        elif event.char == "q":
            self.root.quit()
        elif event.keysym == "Return":
            self.annotation_counter[self.current_index] += 1
            self.user_prompts[self.current_index]["positive"] = []
            self.user_prompts[self.current_index]["negative"] = []
            self.user_prompts[self.current_index]["box"] = None
            self.refresh_display()
        elif event.keysym == "BackSpace":
            self.user_prompts[self.current_index]["positive"] = []
            self.user_prompts[self.current_index]["negative"] = []
            self.user_prompts[self.current_index]["box"] = None
            self.mask_storage[self.current_index][
                self.mask_storage[self.current_index]
                == self.annotation_counter[self.current_index]
            ] = 0
            self.refresh_display()

    def switch_image(self, new_val):
        new_idx = int(float(new_val))
        if new_idx == self.current_index:
            return
        self.current_index = new_idx
        self.image_info.config(
            text=f"Image: {self.current_index +
                           1}/{len(self.image_collection)}"
        )
        self.initialize_segmentation()
        self.refresh_display()

    def export_masks(self):
        if not self.export_dir.exists():
            self.export_dir.mkdir()
        for path_str, mask in zip(self.image_collection.image_paths, self.mask_storage):
            if mask is not None:
                mask_img = Image.fromarray((mask*255.0).astype(np.uint8))
                src_path = pathlib.Path(path_str)
                out_file = str(self.export_dir / src_path.stem) + ".png"
                print(f"Saving {out_file}")
                mask_img.save(out_file)
                print(f"Mask saved to {out_file}")

    def initialize_ui(self):
        self.root = tk.Tk()
        self.root.title("General Segmentation Tool")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(main_frame, width=512, height=512)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<ButtonPress-1>", self.handle_mouse_press)
        self.canvas.bind("<B1-Motion>", self.handle_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.handle_mouse_release)
        self.canvas.bind("<Configure>", self.refresh_display)
        self.root.bind("<Key>", self.process_keystroke)

        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.image_info = tk.Label(
            control_frame,
            text=f"Image: {
                self.current_index + 1}/{len(self.image_collection)}",
        )
        self.image_info.pack(padx=5, pady=5)

        self.image_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=len(self.image_collection) - 1,
            orient=tk.HORIZONTAL,
            command=self.switch_image,
        )
        self.image_slider.set(self.current_index)
        self.image_slider.pack(padx=5, pady=5)

        mode_frame = ttk.Frame(self.root)
        mode_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.mode_label = tk.Label(
            mode_frame,
            text=f"Mode:\n{
                self.input_mode.capitalize()}",
            width=20,
        )
        self.mode_label.pack(padx=5, pady=5)

        self.save_button = ttk.Button(
            self.root, text="Save Masks", command=self.export_masks
        )
        self.save_button.pack(side=tk.TOP, pady=5)

        self.open_button = ttk.Button(
            self.root, text="Open Folder", command=lambda: self.load_images()
        )
        self.open_button.pack(side=tk.TOP, pady=5)

        self.help_button = ttk.Button(self.root, text="Help", command=self.display_help)
        self.help_button.pack(side=tk.TOP, pady=5)

        self.quit_button = ttk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(side=tk.TOP)

        sv_ttk.set_theme("light")

        self.root.protocol("WM_DELETE_WINDOW", sys.exit)
        self.root.mainloop()

        print("Exiting...")

    def display_help(self):
        help_win = tk.Toplevel(self.root)
        help_win.title("Help")
        help_box = tk.Text(help_win, wrap=tk.WORD, width=50, height=20)
        help_box.insert(tk.END, "Instructions:\n\n")
        help_box.insert(tk.END, "1. Use the slider to navigate images.\n")
        help_box.insert(tk.END, "2. Press 'p' for positive point mode.\n")
        help_box.insert(tk.END, "3. Press 'n' for negative point mode.\n")
        help_box.insert(tk.END, "4. Press 'b' for box mode.\n")
        help_box.insert(tk.END, "5. Click and drag to draw a box or add points.\n")
        help_box.insert(tk.END, "6. Press 'r' to reset current annotations.\n")
        help_box.insert(tk.END, "7. Press 'q' to quit the application.\n")
        help_box.config(state=tk.DISABLED)
        help_box.pack(padx=10, pady=10)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <folder_path> <config_key> <device>")
        sys.exit(1)

    folder_path = sys.argv[1]
    config_key = sys.argv[2]
    device = sys.argv[3]
    InteractiveSegmentationTool(folder_path, config_key, device)
