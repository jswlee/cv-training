"""
People detection using YOLO with ROI-based classification
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
from .roi import ROIDetector

logger = logging.getLogger(__name__)

class PeopleDetector:
    """YOLO-based people detection with ROI classification"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize people detector with configuration"""
        self.config = self._load_config(config_path)
        self.models_config = self.config['models']
        self.paths_config = self.config['paths']
        
        # Initialize YOLO model
        self.model = None
        self.roi_detector = ROIDetector(config_path)
        self._load_model()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _load_model(self):
        """Load YOLO model (pre-trained or fine-tuned)"""
        try:
            model_path = self.models_config['yolo_path']
            
            if Path(model_path).exists():
                # Load fine-tuned model
                logger.info(f"Loading fine-tuned YOLO model: {model_path}")
                self.model = YOLO(model_path)
            else:
                # Load pre-trained model
                logger.info("Loading pre-trained YOLOv8n model")
                self.model = YOLO('yolov8n.pt')
                
                # Create models directory if it doesn't exist
                Path(self.models_config['yolo_path']).parent.mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def detect_people(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect people in the image using YOLO
        
        Args:
            image: Input BGR image
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        try:
            if self.model is None:
                raise ValueError("YOLO model not loaded")
            
            # Run inference
            results = self.model(
                image,
                conf=self.models_config['yolo_confidence'],
                iou=self.models_config['yolo_iou'],
                classes=[0]  # Only detect person class (class 0 in COCO)
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': 'person',
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                        }
                        
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} people")
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting people: {e}")
            return []
    
    def classify_detections_by_roi(self, detections: List[Dict[str, Any]], roi_data: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Classify people detections by ROI (water, beach, other)
        
        Args:
            detections: List of people detections
            roi_data: Optional ROI data, will load from file if not provided
            
        Returns:
            Dictionary with detections grouped by ROI zone
        """
        try:
            if roi_data is None:
                roi_data = self.roi_detector.load_roi_zones()
            
            classified = {
                'water': [],
                'beach': [],
                'other': []
            }
            
            for detection in detections:
                center = tuple(detection['center'])
                zone = self.roi_detector.classify_point(center, roi_data)
                
                # Add zone information to detection
                detection['zone'] = zone
                classified[zone].append(detection)
            
            logger.info(f"Classified detections - Water: {len(classified['water'])}, "
                       f"Beach: {len(classified['beach'])}, Other: {len(classified['other'])}")
            
            return classified
            
        except Exception as e:
            logger.error(f"Error classifying detections: {e}")
            return {'water': [], 'beach': [], 'other': []}
    
    def analyze_image(self, image: np.ndarray, roi_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete analysis of people in the image
        
        Args:
            image: Input BGR image
            roi_data: Optional ROI data
            
        Returns:
            Analysis results with counts and detection details
        """
        try:
            # Detect people
            detections = self.detect_people(image)
            
            # Classify by ROI
            classified = self.classify_detections_by_roi(detections, roi_data)
            
            # Create summary
            analysis = {
                'total_people': len(detections),
                'people_in_water': len(classified['water']),
                'people_on_beach': len(classified['beach']),
                'people_other': len(classified['other']),
                'detections': {
                    'water': classified['water'],
                    'beach': classified['beach'],
                    'other': classified['other']
                },
                'confidence_stats': self._calculate_confidence_stats(detections)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                'total_people': 0,
                'people_in_water': 0,
                'people_on_beach': 0,
                'people_other': 0,
                'detections': {'water': [], 'beach': [], 'other': []},
                'confidence_stats': {'mean': 0, 'min': 0, 'max': 0}
            }
    
    def _calculate_confidence_stats(self, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence statistics for detections"""
        if not detections:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0}
        
        confidences = [d['confidence'] for d in detections]
        return {
            'mean': float(np.mean(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
    
    def visualize_detections(self, image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Visualize people detections on the image
        
        Args:
            image: Input BGR image
            analysis: Analysis results from analyze_image
            
        Returns:
            Image with detections drawn
        """
        try:
            vis_image = image.copy()
            
            # Color mapping for different zones
            colors = {
                'water': (255, 0, 0),    # Blue
                'beach': (0, 255, 255),  # Yellow
                'other': (0, 255, 0)     # Green
            }
            
            # Draw detections for each zone
            for zone, detections in analysis['detections'].items():
                color = colors.get(zone, (128, 128, 128))
                
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw center point
                    center = tuple(detection['center'])
                    cv2.circle(vis_image, center, 3, color, -1)
                    
                    # Draw label
                    label = f"{zone}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(vis_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add summary text
            summary = f"Total: {analysis['total_people']} | Water: {analysis['people_in_water']} | Beach: {analysis['people_on_beach']}"
            cv2.putText(vis_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error visualizing detections: {e}")
            return image

def main():
    """Test people detection functionality"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detection.py <image_path>")
        return
    
    image_path = sys.argv[1]
    detector = PeopleDetector()
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Analyze image
        analysis = detector.analyze_image(image)
        
        # Print results
        print(f"Total people detected: {analysis['total_people']}")
        print(f"People in water: {analysis['people_in_water']}")
        print(f"People on beach: {analysis['people_on_beach']}")
        print(f"People in other areas: {analysis['people_other']}")
        print(f"Average confidence: {analysis['confidence_stats']['mean']:.2f}")
        
        # Visualize results
        vis_image = detector.visualize_detections(image, analysis)
        
        # Save visualization
        output_path = "people_detection_result.jpg"
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
