# coding: utf-8

import sys
import pathlib
import glob
import json
from collections import OrderedDict

import numpy as np
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk, colorchooser, simpledialog
from PIL import Image, ImageTk
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sv_ttk


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


class ManualCountDialog(tk.Toplevel):
    def __init__(self, parent, available_classes, current_counts=None):
        super().__init__(parent)
        self.title("Manual Object Count")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.available_classes = available_classes
        self.count_vars = {}
        self.result = None
        
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a label and entry for each class
        ttk.Label(main_frame, text="Specify the count for each object class:", 
                  font=("", 10, "bold")).grid(column=0, row=0, columnspan=2, pady=5, sticky=tk.W)
        
        for i, cls in enumerate(self.available_classes):
            ttk.Label(main_frame, text=f"{cls}:").grid(column=0, row=i+1, sticky=tk.W, padx=5, pady=2)
            
            count_var = tk.StringVar(value=str(current_counts.get(cls, 0)) if current_counts else "0")
            self.count_vars[cls] = count_var
            
            entry = ttk.Entry(main_frame, textvariable=count_var, width=5)
            entry.grid(column=1, row=i+1, sticky=tk.W, padx=5, pady=2)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(column=0, row=len(self.available_classes)+1, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Save", command=self.save_counts).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
        
        # Center the dialog on the parent window
        self.update_idletasks()
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.geometry(f"+{x}+{y}")
        
        self.wait_window(self)
    
    def save_counts(self):
        try:
            # Convert all values to integers
            self.result = {cls: int(var.get()) for cls, var in self.count_vars.items()}
            self.destroy()
        except ValueError:
            # Show error if any value is not an integer
            tk.messagebox.showerror("Invalid Input", "All values must be integers.")
    
    def cancel(self):
        self.result = None
        self.destroy()


class ClassSelectionDialog(tk.Toplevel):
    def __init__(self, parent, available_classes):
        super().__init__(parent)
        self.title("Select Object Class")
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable window close button
        
        self.result = None
        self.available_classes = available_classes
        
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Select the class for this object:", 
                  font=("", 10, "bold")).pack(pady=5)
        
        self.class_var = tk.StringVar(value=available_classes[0])
        
        # Create a dropdown for class selection
        class_combo = ttk.Combobox(main_frame, textvariable=self.class_var, values=available_classes)
        class_combo.pack(padx=5, pady=10, fill=tk.X)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Confirm", command=self.confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.cancel).pack(side=tk.LEFT, padx=5)
        
        # Center the dialog
        self.update_idletasks()
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.geometry(f"+{x}+{y}")
        self.focus_set()
        
        # Make this dialog modal
        self.wait_window(self)
        
    def confirm(self):
        self.result = self.class_var.get()
        self.destroy()
        
    def cancel(self):
        self.result = None
        self.destroy()


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
        self.json_export_dir = pathlib.Path("./annotations")
        self.current_index = 0
        
        # Define available object classes
        self.available_classes = ["person", "car", "building", "card", "animal", "furniture", "other", "token", "dice"]
        self.available_shapes = ["rectangle", "circle", "triangle", "irregular"]
        self.available_values = ["ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "joker", "none"]
        
        # Object metadata storage
        self.object_metadata = {}  # Indexed by image_idx and then object_id
        
        # Manual object counts storage
        self.manual_counts = {}  # Indexed by image_idx
        
        # Current class selection
        self.current_class = self.available_classes[0]
        
        self.load_images(folder_path)

        self.initialize_ui()

    def load_images(self, folder_path=None):
        refresh = False
        if folder_path is None:
            refresh = True
            folder_path = tkinter.filedialog.askdirectory(
                initialdir=self.image_collection.folder_path if hasattr(self, 'image_collection') else ".",
                title="Select an image folder",
            )
            if not folder_path:
                return
        self.export_dir = pathlib.Path("./masks")
        self.json_export_dir = pathlib.Path("./annotations")
        if not self.json_export_dir.exists():
            self.json_export_dir.mkdir(parents=True, exist_ok=True)

        self.image_collection = ImageCollector(folder_path, self.compute_device)

        self.mask_storage = [None] * len(self.image_collection)
        self.annotation_counter = [1] * len(self.image_collection)
        
        # Initialize object metadata and manual counts for each image
        for i in range(len(self.image_collection)):
            if i not in self.object_metadata:
                self.object_metadata[i] = {}
            if i not in self.manual_counts:
                self.manual_counts[i] = {}
                
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
        
        # Store bounding box for the new annotation
        if ann_id not in self.object_metadata[self.current_index]:
            self.object_metadata[self.current_index][ann_id] = {
                "id": ann_id,
                "class": self.current_class,  # Use the pre-selected class
                "shape": self.shape_var.get(),
                "color": self.selected_color.get(),
                "value": self.value_var.get(),
                "bbox": bbox,
                "centroid": None,
                "area": 0
            }
            
            # Calculate centroid and area for the mask
            if masks is not None:
                y_indices, x_indices = np.where(masks == 1.0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    centroid_x = np.mean(x_indices)
                    centroid_y = np.mean(y_indices)
                    self.object_metadata[self.current_index][ann_id]["centroid"] = [centroid_x, centroid_y]
                    self.object_metadata[self.current_index][ann_id]["area"] = len(y_indices)

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
        if (canvas_ratio > img_ratio):
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
        # Check if we need to select a class first
        if not hasattr(self, 'current_class') or self.current_class is None:
            if not self.select_class_before_annotation():
                return  # User cancelled class selection
            
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
        # For mode changes, update the current class first
        if event.char in ["p", "n", "b"]:
            if not self.select_class_before_annotation():
                return
                
        if event.char == "p":
            self.input_mode = "point"
            self.point_type = "positive"
            self.mode_label.config(text=f"Mode:\nPoint (Positive)\nClass: {self.current_class}")
        elif event.char == "n":
            self.input_mode = "point"
            self.point_type = "negative"
            self.mode_label.config(text=f"Mode:\nPoint (Negative)\nClass: {self.current_class}")
        elif event.char == "b":
            self.input_mode = "box"
            self.mode_label.config(text=f"Mode:\nBox\nClass: {self.current_class}")
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
            # Update object metadata before creating a new annotation
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
        self.update_count_indicator()  # Update the manual count indicator
        self.refresh_display()

    def select_color(self):
        color = colorchooser.askcolor(title="Select a color for the object")
        if color[1]: # color is [RGB tuple, hex string]
            self.selected_color.set(color[1])
            self.color_button.configure(background=color[1])

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
        
        # Export JSON annotations
        self.export_json_annotations()

    def export_json_annotations(self):
        """Export annotations to JSON files as specified in requirements"""
        if not self.json_export_dir.exists():
            self.json_export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the three types of JSON files
        self.export_detailed_annotations()
        self.export_object_list()
        self.export_object_counts()
        
    def export_detailed_annotations(self):
        """Export detailed annotations with positions and relations"""
        for img_idx, img_path in enumerate(self.image_collection.image_paths):
            if img_idx not in self.object_metadata or not self.object_metadata[img_idx]:
                continue
                
            img_filename = pathlib.Path(img_path).stem
            annotations = []
            
            # Calculate relative positions
            objects = list(self.object_metadata[img_idx].values())
            for obj in objects:
                obj_data = {
                    "id": obj["id"],
                    "class": obj["class"],
                    "shape": obj["shape"],
                    "color": obj["color"],
                    "value": obj["value"],
                    "absolute_position": {
                        "bbox": obj["bbox"],
                        "centroid": obj["centroid"],
                        "area": obj["area"]
                    },
                    "relative_positions": []
                }
                
                # Calculate relative positions to other objects
                if obj["centroid"] is not None:
                    for other_obj in objects:
                        if other_obj["id"] != obj["id"] and other_obj["centroid"] is not None:
                            dx = other_obj["centroid"][0] - obj["centroid"][0]
                            dy = other_obj["centroid"][1] - obj["centroid"][1]
                            distance = np.sqrt(dx*dx + dy*dy)
                            
                            # Determine direction (N, NE, E, SE, S, SW, W, NW)
                            angle = np.arctan2(dy, dx) * 180 / np.pi
                            direction = ""
                            if -22.5 <= angle < 22.5:
                                direction = "E"
                            elif 22.5 <= angle < 67.5:
                                direction = "SE"
                            elif 67.5 <= angle < 112.5:
                                direction = "S"
                            elif 112.5 <= angle < 157.5:
                                direction = "SW"
                            elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
                                direction = "W"
                            elif -157.5 <= angle < -112.5:
                                direction = "NW"
                            elif -112.5 <= angle < -67.5:
                                direction = "N"
                            elif -67.5 <= angle < -22.5:
                                direction = "NE"
                                
                            obj_data["relative_positions"].append({
                                "to_object_id": other_obj["id"],
                                "to_object_class": other_obj["class"],
                                "distance": float(distance),
                                "direction": direction
                            })
                            
                annotations.append(obj_data)
            
            # Save to file
            with open(f"{self.json_export_dir}/{img_filename}_detailed.json", 'w') as f:
                json.dump({"image": img_filename, "annotations": annotations}, f, indent=2)
            print(f"Detailed annotations saved to {self.json_export_dir}/{img_filename}_detailed.json")
            
    def export_object_list(self):
        """Export list of all objects in each image"""
        for img_idx, img_path in enumerate(self.image_collection.image_paths):
            if img_idx not in self.object_metadata or not self.object_metadata[img_idx]:
                continue
                
            img_filename = pathlib.Path(img_path).stem
            objects_list = []
            
            for obj_id, obj_data in self.object_metadata[img_idx].items():
                objects_list.append({
                    "id": obj_data["id"],
                    "class": obj_data["class"],
                    "shape": obj_data["shape"],
                    "color": obj_data["color"],
                    "value": obj_data["value"]
                })
            
            # Save to file
            with open(f"{self.json_export_dir}/{img_filename}_objects.json", 'w') as f:
                json.dump({
                    "image": img_filename,
                    "object_count": len(objects_list),
                    "objects": objects_list
                }, f, indent=2)
            print(f"Object list saved to {self.json_export_dir}/{img_filename}_objects.json")
            
    def export_object_counts(self):
        """Export count of objects by class for each image"""
        for img_idx, img_path in enumerate(self.image_collection.image_paths):
            img_filename = pathlib.Path(img_path).stem
            
            # Use manual counts if available, otherwise calculate from segmentations
            if img_idx in self.manual_counts and any(self.manual_counts[img_idx].values()):
                class_counts = self.manual_counts[img_idx].copy()
                total_count = sum(class_counts.values())
                is_manual = True
                
                # Create a list of objects with IDs for each class
                objects_by_class = {}
                current_id = 1
                for class_name, count in class_counts.items():
                    objects_by_class[class_name] = []
                    for i in range(count):
                        objects_by_class[class_name].append({
                            "id": current_id,
                            "class": class_name
                        })
                        current_id += 1
            else:
                if img_idx not in self.object_metadata or not self.object_metadata[img_idx]:
                    continue
                    
                class_counts = {}
                objects_by_class = {}
                
                # Group existing objects by class
                for obj_id, obj_data in self.object_metadata[img_idx].items():
                    obj_class = obj_data["class"]
                    if obj_class in class_counts:
                        class_counts[obj_class] += 1
                    else:
                        class_counts[obj_class] = 1
                        objects_by_class[obj_class] = []
                        
                    # Add this object to its class group
                    objects_by_class[obj_class].append({
                        "id": obj_id,
                        "class": obj_class
                    })
                    
                total_count = len(self.object_metadata[img_idx])
                is_manual = False
                
            # Ensure all classes are in the counts
            for class_name in self.available_classes:
                if class_name not in class_counts:
                    class_counts[class_name] = 0
            
            # Save to file to json (image_name_counts.json)
            with open(f"{self.json_export_dir}/{img_filename}_counts.json", 'w') as f:
                json.dump({
                    "image": img_filename,
                    "total_objects": total_count,
                    "class_counts": class_counts,
                    "objects_by_class": objects_by_class,
                    "is_manual_count": is_manual
                }, f, indent=2)
            print(f"Object counts saved to {self.json_export_dir}/{img_filename}_counts.json")

    def open_manual_count_dialog(self):
        """Open a dialog to manually set the object counts for the current image"""
        current_counts = self.manual_counts.get(self.current_index, {})
        dialog = ManualCountDialog(self.root, self.available_classes, current_counts)
        
        if dialog.result is not None:
            self.manual_counts[self.current_index] = dialog.result
            self.update_count_indicator()
    
    def update_count_indicator(self):
        """Update the UI to show if manual counts are set for this image"""
        if self.current_index in self.manual_counts and any(self.manual_counts[self.current_index].values()):
            self.manual_count_indicator.config(foreground="green", text="Manual counts: Set ✓")
        else:
            self.manual_count_indicator.config(foreground="red", text="Manual counts: Not set")

    def select_class_before_annotation(self):
        """Open a dialog to select the class before starting an annotation"""
        dialog = ClassSelectionDialog(self.root, self.available_classes)
        if dialog.result:
            self.current_class = dialog.result
            self.class_var.set(dialog.result)
            return True
        return False

    def initialize_ui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Annotation Tool")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(main_frame, width=512, height=512)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<ButtonPress-1>", self.handle_mouse_press)
        self.canvas.bind("<B1-Motion>", self.handle_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.handle_mouse_release)
        self.canvas.bind("<Configure>", self.refresh_display)
        self.root.bind("<Key>", self.process_keystroke)

        # Create the left control panel (to modify)
        control_frame = ttk.Frame(self.root, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Image navigation control 
        nav_frame = ttk.LabelFrame(control_frame, text="Image Navigation")
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.image_info = ttk.Label(
            nav_frame,
            text=f"Image: {self.current_index + 1}/{len(self.image_collection)}",
        )
        self.image_info.pack(padx=5, pady=5)

        self.image_slider = ttk.Scale(
            nav_frame,
            from_=0,
            to=len(self.image_collection) - 1,
            orient=tk.HORIZONTAL,
            command=self.switch_image,
        )
        self.image_slider.set(self.current_index)
        self.image_slider.pack(padx=5, pady=5, fill=tk.X)
        
        # Input mode controls
        mode_frame = ttk.LabelFrame(control_frame, text="Input Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)

        self.mode_label = ttk.Label(
            mode_frame,
            text=f"Current: {self.input_mode.capitalize()}\nClass: {self.current_class}",
            width=20,
        )
        self.mode_label.pack(padx=5, pady=5)
        
        set_class_button = ttk.Button(
            mode_frame,
            text="Set Object Class",
            command=self.select_class_before_annotation
        )
        set_class_button.pack(fill=tk.X, padx=5, pady=2)
        
        mode_buttons = ttk.Frame(mode_frame)
        mode_buttons.pack(fill=tk.X)
        
        ttk.Button(mode_buttons, text="Box (b)", command=lambda: self.process_keystroke(type('obj', (), {'char': 'b'})())).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(mode_buttons, text="Pos Point (p)", command=lambda: self.process_keystroke(type('obj', (), {'char': 'p'})())).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(mode_buttons, text="Neg Point (n)", command=lambda: self.process_keystroke(type('obj', (), {'char': 'n'})())).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # Object attributes frame
        attr_frame = ttk.LabelFrame(control_frame, text="Object Attributes")
        attr_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Class selection
        ttk.Label(attr_frame, text="Class:").pack(anchor=tk.W, padx=5, pady=2)
        self.class_var = tk.StringVar(value=self.available_classes[0])
        class_combo = ttk.Combobox(attr_frame, textvariable=self.class_var, values=self.available_classes)
        class_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Shape selection
        ttk.Label(attr_frame, text="Shape:").pack(anchor=tk.W, padx=5, pady=2)
        self.shape_var = tk.StringVar(value=self.available_shapes[0])
        shape_combo = ttk.Combobox(attr_frame, textvariable=self.shape_var, values=self.available_shapes)
        shape_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Color selection
        ttk.Label(attr_frame, text="Color:").pack(anchor=tk.W, padx=5, pady=2)
        color_frame = ttk.Frame(attr_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.selected_color = tk.StringVar(value="#FF0000")  # Default red
        self.color_button = tk.Button(
            color_frame, 
            text="Select Color",
            background=self.selected_color.get(),
            command=self.select_color
        )
        self.color_button.pack(fill=tk.X)
        
        # Value selection (for cards)
        ttk.Label(attr_frame, text="Value (for cards):").pack(anchor=tk.W, padx=5, pady=2)
        self.value_var = tk.StringVar(value=self.available_values[-1])  # Default to "none"
        value_combo = ttk.Combobox(attr_frame, textvariable=self.value_var, values=self.available_values)
        value_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Manual count frame
        count_frame = ttk.LabelFrame(control_frame, text="Manual Object Count")
        count_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.manual_count_indicator = ttk.Label(
            count_frame, 
            text="Manual counts: Not set",
            foreground="red"
        )
        self.manual_count_indicator.pack(padx=5, pady=2, anchor=tk.W)
        
        ttk.Button(
            count_frame,
            text="Set Manual Counts",
            command=self.open_manual_count_dialog
        ).pack(fill=tk.X, padx=5, pady=5)
        
        # Action buttons
        actions_frame = ttk.LabelFrame(control_frame, text="Actions")
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_button = ttk.Button(
            actions_frame, text="Export All", command=self.export_masks
        )
        self.save_button.pack(fill=tk.X, padx=5, pady=2)

        self.open_button = ttk.Button(
            actions_frame, text="Open Folder", command=lambda: self.load_images()
        )
        self.open_button.pack(fill=tk.X, padx=5, pady=2)

        self.help_button = ttk.Button(actions_frame, text="Help", command=self.display_help)
        self.help_button.pack(fill=tk.X, padx=5, pady=2)

        self.quit_button = ttk.Button(actions_frame, text="Quit", command=self.root.quit)
        self.quit_button.pack(fill=tk.X, padx=5, pady=2)

        sv_ttk.set_theme("light")

        # Update the count indicateur 
        self.update_count_indicator()

        self.root.protocol("WM_DELETE_WINDOW", sys.exit)
        self.root.mainloop()

        print("Exiting...")

    def display_help(self):
        help_win = tk.Toplevel(self.root)
        help_win.title("Help")
        help_box = tk.Text(help_win, wrap=tk.WORD, width=50, height=22)
        help_box.insert(tk.END, "Instructions:\n\n")
        help_box.insert(tk.END, "1. Use the slider to navigate images.\n")
        help_box.insert(tk.END, "2. Press 'p' for positive point mode.\n")
        help_box.insert(tk.END, "3. Press 'n' for negative point mode.\n")
        help_box.insert(tk.END, "4. Press 'b' for box mode.\n")
        help_box.insert(tk.END, "5. Click and drag to draw a box or add points.\n")
        help_box.insert(tk.END, "6. Press 'r' to reset current annotations.\n")
        help_box.insert(tk.END, "7. Press 'Enter' to confirm an object and create a new one.\n")
        help_box.insert(tk.END, "8. Before confirming an object, set its class, shape, color and value.\n")
        help_box.insert(tk.END, "9. Use 'Set Manual Counts' to specify object counts when automatic detection is inaccurate.\n")
        help_box.insert(tk.END, "10. Press 'Export All' to save masks and JSON annotations.\n")
        help_box.insert(tk.END, "11. Press 'q' to quit the application.\n")
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
