from ultralytics import YOLO
from typing import List, Tuple, Dict
import torch

class YOLOPredictor:
    def __init__(self):
        self.model = None
        self.class_names = []

    def load_model(self, model_path: str) -> Dict[int, str]:
        """
        Load a YOLO model from the given path
        Args:
            model_path: Path to the .pt model file
        Returns:
            Dictionary mapping class indices to class names
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            return self.class_names
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def predict(self, image_path: str) -> List[Tuple[int, float, float, float, float]]:
        """
        Run inference on an image and return detections
        Args:
            image_path: Path to the image file
        Returns:
            List of tuples (class_id, x1, y1, x2, y2) where coordinates are in pixels
        """
        if not self.model:
            raise Exception("No model loaded. Please load a model first.")

        try:
            # Run inference
            results = self.model(image_path)
            result = results[0]
            
            detections = []
            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                
                # Convert to integers for pixel coordinates
                detections.append((
                    class_id,
                    int(x1), int(y1), int(x2), int(y2)
                ))
            
            return detections
        except Exception as e:
            raise Exception(f"Failed to run inference: {str(e)}")

    @property
    def has_model(self) -> bool:
        """Check if a model is loaded"""
        return self.model is not None

    def get_class_names(self) -> Dict[int, str]:
        """Get the class names from the loaded model"""
        if not self.model:
            raise Exception("No model loaded. Please load a model first.")
        return self.class_names 