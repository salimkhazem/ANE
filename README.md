# Advanced Annotation Tool (Gaime case)

This is an enhanced annotation tool built for object segmentation and metadata capture. It allows users to segment objects in images and annotate them with additional information like class, shape, color, and value.


![Gaime](https://github.com/user-attachments/assets/0723caec-5a28-4664-9b49-976edfefb367)


## Features

- Interactive segmentation using SAM2 models
- Object classification with customizable classes
- Shape, color, and value annotations for objects
- Manual object counting when automatic segmentation is insufficient
- AI-generated image legends in French using Azure OpenAI
- JSON export for detailed annotations including:
  - Object metadata (class, shape, color, value)
  - Absolute positions (bounding box, centroid)
  - Relative positions between objects
  - Object counts by class with unique IDs
  - Complete object lists
  - Image legends/descriptions

## Installation

```bash
# Clone the repository
git clone <repository_url>

# Navigate to the directory
cd ANE

# Install dependencies
pip install -r requirements.txt
```

## Usage

You can run the application in two ways:

### Option 1: Direct Python Command

```bash
python gui.py <folder_path> <config_key> <device>
```

Where:
- `folder_path`: Path to the folder containing images
- `config_key`: Model configuration (e.g., "sam2.1-hiera-tiny")
- `device`: Computation device ("cuda" or "cpu")

Example:
```bash
python gui.py ./images sam2.1-hiera-tiny cpu
```

### Option 2: Using the run.sh Script

For convenience, especially when using the Azure OpenAI features, you can use the run.sh script:

1. Edit the run.sh script to include your Azure OpenAI API key:
   ```bash
   # Open the script
   vim run.sh
   
   # Ensure it contains these lines, replacing with your actual API key:
   export AZURE_OPENAI_API_KEY="your-api-key-here"
   python gui.py <folder_path> <config_key> <device>
   ```

2. Make the script executable and run it:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

> **Important**: You must set your Azure OpenAI API key in run.sh to use the legend generation feature. Make sure the Python command in run.sh points to the correct image folder, model configuration, and device.

## Interface Controls

- **Set Object Class**: Select the class before starting annotation
- **Box Mode (b)**: Draw bounding boxes around objects
- **Positive Point Mode (p)**: Add positive points to include in segmentation
- **Negative Point Mode (n)**: Add negative points to exclude from segmentation
- **Enter**: Confirm current object and create a new one
- **r**: Reset current annotations
- **Left/Right Arrow Keys**: Navigate between images
- **Set Manual Counts**: Open a dialog to manually specify the number of objects by class
- **Edit Legend**: Create or edit AI-generated image descriptions
- **Export All**: Save masks and JSON annotations

## Class Selection Workflow

1. Before annotating an object, you'll be prompted to select its class
2. Choose a class from the dropdown menu (e.g., "card", "token", "dice")
3. Proceed with the annotation using box or point mode
4. Each object gets assigned a unique ID across all classes

## Manual Object Counting

When automatic segmentation doesn't accurately detect all objects (for example with small or overlapping objects), you can:

1. Click the "Set Manual Counts" button
2. Enter the number of objects for each class
3. Save the counts

These manual counts will override automatic detection counts in the JSON exports.

## Image Legends with Azure OpenAI

The tool includes AI-powered image description generation:

1. Click "Edit Legend" to open the legend editor
2. Click "Generate Legend" to create a description using Azure OpenAI
3. Edit the text as needed
4. Click "Save Legend" to store it with the image data

The generated legend will be included in the JSON output and typically describes the image in French, with text like:
```
Dans cette image, on peut voir 9 cartes et 2 jetons. Les jetons se trouvent à droite des cartes.
```

### Azure OpenAI Configuration

To use the legend generation feature, you need to set your Azure OpenAI API key. The easiest way is to use the run.sh script as described above. Alternatively, you can set it as an environment variable before running the application:

```bash
export AZURE_OPENAI_API_KEY="your-api-key-here"
python gui.py ./images sam2.1-hiera-tiny cpu
```

## JSON Export Format

The tool generates three types of JSON files:

1. **Detailed Annotations** (`image_detailed.json`):
   - Object metadata (id, class, shape, color, value)
   - Absolute positions (bbox, centroid, area)
   - Relative positions to other objects (distance, direction)

2. **Object List** (`image_objects.json`):
   - List of all objects in the image with unique IDs
   - Basic metadata for each object
   - Total object count

3. **Object Counts** (`image_counts.json`):
   - Total number of objects across all classes
   - Count by class
   - List of objects with IDs by class
   - Whether the count is manual or automatic
   - Image legend/description

## Example JSON Output

For an image with 9 cards and 2 tokens:

```json
{
  "image": "game_table",
  "total_objects": 11,
  "class_counts": {
    "card": 9,
    "token": 2,
    "dice": 0
  },
  "objects_by_class": {
    "card": [
      {"id": 1, "class": "card"},
      {"id": 2, "class": "card"},
      {"id": 3, "class": "card"},
      {"id": 4, "class": "card"},
      {"id": 5, "class": "card"},
      {"id": 6, "class": "card"},
      {"id": 7, "class": "card"},
      {"id": 8, "class": "card"},
      {"id": 9, "class": "card"}
    ],
    "token": [
      {"id": 10, "class": "token"},
      {"id": 11, "class": "token"}
    ],
    "dice": []
  },
  "is_manual_count": true,
  "legend": "Dans cette image, on peut voir 9 cartes et 2 jetons. Les jetons se trouvent à droite des cartes."
}




