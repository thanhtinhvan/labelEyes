# YOLO Annotation Tool

A simple annotation tool for creating YOLO format annotations using Python and PyQt5.

## Features

- Load and browse through images from a folder
- Draw bounding boxes on images
- Select object classes from a predefined list
- Save annotations in YOLO format
- Navigate between images using next/previous buttons
- View all images in the folder in a list

## Requirements

- Python 3.6+
- PyQt5
- Pillow
- numpy

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Click "Load Image Folder" to select a folder containing your images
3. Select an object class from the dropdown menu
4. Draw bounding boxes by clicking and dragging on the image
5. Click "Save Annotations" to save the annotations in YOLO format
6. Use "Next Image" and "Previous Image" buttons to navigate through the images

## YOLO Format

The tool saves annotations in YOLO format, where each line represents:
```
<object-class> <x-center> <y-center> <width> <height>
```
- All values are normalized between 0 and 1
- Object class is an integer starting from 0
- (x-center, y-center) represents the center of the bounding box
- (width, height) represents the width and height of the bounding box

## Keyboard Shortcuts

- Next Image: Right Arrow
- Previous Image: Left Arrow
- Save: Ctrl+S

## Customizing Classes

To modify the available classes, edit the `classes` list in the `AnnotationTool` class initialization. 