import torch
import numpy as np
from ultralytics import YOLOE
from typing import Tuple, List, Union
import cv2
import os
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

NAMES = ["ARROW", "CIRCLE", "MARKER"]

class YOLOEPredictor:
    def __init__(self, model_path: str = "jameslahm/yoloe-v8l-seg", device: str = "cuda:0"):
        """Initialize YOLOE predictor with visual prompting capability.
        
        Args:
            model_path: Path to YOLOE model or model name from HuggingFace
            device: Device to run inference on ('cuda:0' or 'cpu')
        """
        self.device = device
        # Load YOLOE model
        print(f"Loading YOLOE model from {model_path}")
        self.model = YOLOE(model_path)
        self.model.to(device)
        
        # Store current prompt info
        self.current_prompt_box = None
        self.current_prompt_class = None
        
    def set_visual_prompt(self, bboxes: List[float], class_name: str):
        """Set the visual prompt for subsequent predictions.
        
        Args:
            bboxes: list of coordinates [[x1, y1, x2, y2], [x1, y1, x2, y2], ...] of prompt box
            class_name: Class name for the prompt
        """
        #convert prompt_box to np.array dtype=np.float64
        print("PROMPT BOX: ", bboxes, "CLASS NAME: ", class_name)
        prompt_box = np.array([bboxes], dtype=np.float64)
        #convert class_name to np array np.int32
        class_name_idx = np.array([NAMES.index(box['label'])] for box in bboxes], dtype=np.int32)
        self.current_prompt_box = dict(bboxes=prompt_box, cls=class_name)
        # self.current_prompt_box = prompt_box
        # self.current_prompt_class = class_name
        
    def predict(self, image: Union[str, np.ndarray], conf_threshold: float = 0.5, is_prompt: bool = False) -> List[dict]:
        """Predict objects in image using current visual prompt.
        
        Args:
            image: Input image (numpy array or path to image)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of dictionaries containing detection results with format:
            {
                'box': [x1, y1, x2, y2],  # Normalized coordinates
                'confidence': float,
                'class_name': str
            }
        """
        if self.current_prompt_box is None:# or self.current_prompt_class is None:
            raise ValueError("Visual prompt not set. Call set_visual_prompt first.")
            
        # Prepare image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        # Run prediction with visual prompting
        if is_prompt:
            print("CURRENT PROMPT BOX: ", self.current_prompt_box)
            results = self.model.predict(
                image,
                prompts=self.current_prompt_box,
                predictor=YOLOEVPSegPredictor, return_vpe=True, conf=conf_threshold, agnostic_nms=True
            )
            self.model.set_classes([0], self.model.predictor.vpe)
            self.model.predictor = None
        else:
            results = self.model.predict(
                image,
                conf=conf_threshold
            )
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box, conf in zip(boxes.xyxyn, boxes.conf):
                detections.append({
                    'box': box.tolist(),
                    'confidence': float(conf),
                    'class_name': self.current_prompt_class
                })
                
        return detections

    def visualize(self, image: Union[str, np.ndarray], detections: List[dict]) -> np.ndarray:
        """Visualize detection results on image.
        
        Args:
            image: Input image (numpy array or path to image)
            detections: List of detection dictionaries from predict()
            
        Returns:
            Image with visualized detections
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        img_h, img_w = image.shape[:2]
        
        # Draw detections
        for det in detections:
            box = det['box']
            conf = det['confidence']
            
            # Convert normalized coordinates to pixel coordinates
            x1, y1, x2, y2 = [
                int(box[0] * img_w),
                int(box[1] * img_h),
                int(box[2] * img_w),
                int(box[3] * img_h)
            ]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det['class_name']} {conf:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return image 

# Add new function for drawing box
def draw_box(event, x, y, flags, param):
    img = param['image'].copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        param['drawing'] = True
        param['top_left'] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['drawing']:
            cv2.rectangle(img, param['top_left'], (x, y), (0, 255, 0), 2)
            cv2.imshow('Draw Box', img)
    elif event == cv2.EVENT_LBUTTONUP:
        param['drawing'] = False
        param['bottom_right'] = (x, y)
        cv2.rectangle(img, param['top_left'], (x, y), (0, 255, 0), 2)
        cv2.imshow('Draw Box', img)
        param['image'] = img.copy()

if __name__ == "__main__":
    # Create result directory if it doesn't exist
    os.makedirs("result", exist_ok=True)
    
    # Get list of images from test_image folder
    image_folder = "test_image"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in test_image folder!")
        exit()
    
    # Initialize predictor
    predictor = YOLOEPredictor(
        model_path="yoloe-v8m-seg",
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # Process first image for drawing box
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    
    # Setup mouse callback parameters
    param = {
        'image': first_image.copy(),
        'drawing': False,
        'top_left': None,
        'bottom_right': None
    }
    
    # Create window and set mouse callback
    cv2.namedWindow('Draw Box')
    cv2.setMouseCallback('Draw Box', draw_box, param)
    
    print("Draw a bounding box on the image and press 'Enter' when done")
    print("Press 'r' to reset the box")
    print("Press 'q' to quit")
    
    while True:
        cv2.imshow('Draw Box', param['image'])
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if param['top_left'] and param['bottom_right']:
                break
        elif key == ord('r'):  # Reset
            param['image'] = first_image.copy()
            param['top_left'] = None
            param['bottom_right'] = None
        elif key == ord('q'):  # Quit
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()
    # Get class name from user
    class_name = input("Enter the class name for the selected object: ")
    
    # Convert box coordinates to normalized format
    img_h, img_w = first_image.shape[:2]
    x1 = min(param['top_left'][0], param['bottom_right'][0])# / img_w
    y1 = min(param['top_left'][1], param['bottom_right'][1])# / img_h
    x2 = max(param['top_left'][0], param['bottom_right'][0])# / img_w
    y2 = max(param['top_left'][1], param['bottom_right'][1])# / img_h
    prompt_box = [x1, y1, x2, y2]
    
    # Set the visual prompt
    predictor.set_visual_prompt(prompt_box, class_name)
    predictor.predict(first_image_path, is_prompt=True)
    
    # Process all images (including first one)
    for img_file in image_files:
        image_path = os.path.join(image_folder, img_file)
        print("IMAGE PATH: ", image_path)
        if True:#try:
            # Run prediction
            detections = predictor.predict(
                image=image_path,
                conf_threshold=0.5,
                is_prompt=False
            )
            
            # Visualize results
            result_image = predictor.visualize(image_path, detections)
            
            # Save result
            output_path = os.path.join("result", f"result_{img_file}")
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            # Display results
            print(f"\nProcessing {img_file}:")
            print(f"Found {len(detections)} {class_name}(s)")
            for i, det in enumerate(detections, 1):
                print(f"Detection {i}:")
                print(f"  Box: {det['box']}")
                print(f"  Confidence: {det['confidence']:.3f}")
            
            # Show result (optional)
            cv2.imshow("Result", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0)
            if key == ord('q'):
                continue
                
        # except Exception as e:
        #     print(f"Error processing {img_file}: {e}")
    
    cv2.destroyAllWindows()
    print("\nAll results saved in 'result' folder") 