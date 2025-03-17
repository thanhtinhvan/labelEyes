# AI-Powered YOLO Annotation Tool

A smart annotation tool that uses YOLOE (YOLO Everything) to accelerate the labeling process. Label one image, and let AI help you label the rest!

## Key Features

- **Smart Labeling with YOLOE**:
  - Draw bounding box on the first image
  - AI automatically detects similar objects in subsequent images
  - Learns from your initial annotation to speed up the labeling process

- **Interactive Annotation**:
  - Load and browse through images from a folder
  - Draw bounding boxes with click-and-drag
  - Visual prompt-based object detection
  - Review and adjust AI-generated annotations

- **Efficient Workflow**:
  - Start by annotating just one example
  - AI generates labels for remaining images
  - Review and correct AI predictions if needed
  - Save annotations in YOLO format

## Requirements

- Python 3.8+
- PyQt5
- YOLOE dependencies
- OpenCV
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

2. **First Image Annotation**:
   - Load your image folder
   - Draw a bounding box around your object of interest
   - Enter the class name for the object
   - This becomes the visual prompt for YOLOE

3. **Automatic Labeling**:
   - YOLOE uses your first annotation as a reference
   - Automatically detects and labels similar objects
   - Results are shown for review

4. **Review and Save**:
   - Review AI-generated labels
   - Adjust or correct if needed
   - Save final annotations in YOLO format

## YOLO Format

The tool saves annotations in YOLO format:
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
- Reset Box: R
- Quit: Q

## Roadmap

- [ ] Auto-training integration
  - Automatically fine-tune YOLOE on your dataset
  - Improve detection accuracy with more examples
  - Export trained model for inference
- [ ] Batch processing capabilities
- [ ] Multi-class support in single session
- [ ] Export to various formats
- [ ] Active learning integration

## How It Works

1. **Visual Prompting**:
   - Your first annotation serves as a visual prompt
   - YOLOE learns what to look for from this example

2. **AI Detection**:
   - YOLOE processes subsequent images
   - Uses visual similarity to find matching objects
   - Generates bounding box predictions

3. **Future: Auto Training**:
   - Collect verified annotations
   - Fine-tune YOLOE on your specific dataset
   - Improve detection accuracy over time

## Contributing

Contributions are welcome! Please feel free to submit pull requests. 