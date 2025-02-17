# Annotation Tool using SAM2

## Overview
This project provides an image annotation tool for segmentation leveraging the **SAM2 (Segment Anything Model v2)** framework. It includes both a GUI-based interactive segmentation tool (`gui.py`) and a web-based application built with **Streamlit** (`stapp.py`).

![Example](https://github.com/user-attachments/assets/d5fa944e-fefa-4092-b24c-1efe6ca4c977)


## Features
- **Interactive Segmentation (GUI-based)**
  - Load images from a directory
  - Annotate images with positive/negative points or bounding boxes
  - Apply segmentation using the SAM2 model
  - Save segmented masks
  
- **Web-based Segmentation (Streamlit App)**
  - Select image directories dynamically
  - Choose different segmentation models
  - Perform segmentation on selected images
  - Save segmentation results

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Dependencies
Install required packages using:
```bash
pip install -r requirements.txt
```
## Download the checkpoints 

For running SAM2, you need to manually download the pytorch checkpoints.

To get the checkpoints, meta is giving us a bash script to run :

```bash
mkdir checkpoints
cd checkpoints
wget https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/checkpoints/download_ckpts.sh
bash download_ckpts.sh
```
### Additional Requirements
- **SAM2 Model Checkpoints**
  - Ensure model weights are placed in the `checkpoints/` directory.
  
## Usage
### GUI-based Segmentation
Run the GUI tool with:
```bash
python gui.py <image_directory> <model_name>
```
Example:
```bash
python gui.py /path/to/images sam2.1-hiera-tiny
```

### Web-based Segmentation (Streamlit App)
If you are using ssh, you have to ssh into your server with L option: 
```bash 
ssh -L PORT:127.0.0.1:PORT username@ip_address
```
for example if you want to use 8082, you just need to replace PORT above with 8082. 

After that you can launch the Streamlit app with:
```bash
streamlit run stapp.py --server.address 127.0.0.1 --server.port 8082
```

## Project Structure
```
.
├── gui.py              # GUI-based segmentation tool using Tkinter
├── stapp.py            # Streamlit web app for segmentation
├── requirements.txt    # Dependencies list
├── checkpoints/        # Directory for model weights
└── segmented_outputs/  # Directory where segmentation results are saved
```

## Model Selection
The following models are supported:
- `sam2.1-hiera-tiny`
- `sam2.1-hiera-small`
- `sam2.1-hiera-base-plus`
- `sam2.1-hiera-large`

## How It Works
1. **Load Images** - Select an image directory
2. **Choose Model** - Pick a segmentation model
3. **Apply Segmentation** - Use annotations or bounding boxes to guide the segmentation
4. **View & Save Results** - Preview and store segmentation masks



