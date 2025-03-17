import torch
import numpy as np
from ultralytics import YOLOE
from typing import Tuple, List, Union
import cv2
import os

class YOLOEPredictor:
    def __init__(self, model_path: str = "jameslahm/yoloe-v8m-seg", device: str = "cuda:0"):
        """Initialize YOLOE predictor with visual prompting capability.
        
        Args:
            model_path: Path to YOLOE model or model name from HuggingFace
            device: Device to run inference on ('cuda:0' or 'cpu')
        """
        self.device = device
        # Load YOLOE model
        self.model = YOLOE.from_pretrained(model_path)
        self.model.to(device)
        
        # Store current prompt info
        self.current_prompt_box = None
        self.current_prompt_class = None
        
    def set_visual_prompt(self, prompt_box: List[float], class_name: str):
        """Set the visual prompt for subsequent predictions.
        
        Args:
            prompt_box: Normalized coordinates [x1, y1, x2, y2] of prompt box
            class_name: Class name for the prompt
        """
        self.current_prompt_box = prompt_box
        self.current_prompt_class = class_name
        
    def predict(self, image: Union[str, np.ndarray], conf_threshold: float = 0.5) -> List[dict]:
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
        if self.current_prompt_box is None or self.current_prompt_class is None:
            raise ValueError("Visual prompt not set. Call set_visual_prompt first.")
            
        # Prepare image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Create prompt dictionary
        prompt = {
            'box': self.current_prompt_box,
            'class': self.current_prompt_class
        }
            
        # Run prediction with visual prompting
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False,
            prompt=prompt  # Pass prompt as a dictionary
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

    # ... rest of the code remains the same ... 