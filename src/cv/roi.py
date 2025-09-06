"""
Computer vision-based ROI (Region of Interest) detection for beach scenes
Automatically detects water and beach areas using color segmentation and morphological operations
"""
import cv2
import numpy as np
import json
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ROIDetector:
    """Computer vision-based ROI detector for beach scenes"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ROI detector with configuration"""
        self.config = self._load_config(config_path)
        self.roi_config = self.config['roi']
        self.paths_config = self.config['paths']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def detect_water_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect water regions in the image using color segmentation
        
        Args:
            image: Input BGR image
            
        Returns:
            List of contours representing water regions
        """
        try:
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for water colors (multiple blue ranges)
            water_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for color_range in self.roi_config['water_color_ranges']:
                lower = np.array(color_range[0])
                upper = np.array(color_range[1])
                mask = cv2.inRange(hsv, lower, upper)
                water_mask = cv2.bitwise_or(water_mask, mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = self.roi_config['min_area']
            water_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            logger.info(f"Detected {len(water_contours)} water regions")
            return water_contours
            
        except Exception as e:
            logger.error(f"Error detecting water regions: {e}")
            return []
    
    def detect_beach_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect beach/sand regions in the image
        
        Args:
            image: Input BGR image
            
        Returns:
            List of contours representing beach regions
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for beach colors (sandy/brown ranges)
            beach_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for color_range in self.roi_config['beach_color_ranges']:
                lower = np.array(color_range[0])
                upper = np.array(color_range[1])
                mask = cv2.inRange(hsv, lower, upper)
                beach_mask = cv2.bitwise_or(beach_mask, mask)
            
            # Apply morphological operations
            kernel = np.ones((7, 7), np.uint8)
            beach_mask = cv2.morphologyEx(beach_mask, cv2.MORPH_CLOSE, kernel)
            beach_mask = cv2.morphologyEx(beach_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(beach_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = self.roi_config['min_area']
            beach_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            logger.info(f"Detected {len(beach_contours)} beach regions")
            return beach_contours
            
        except Exception as e:
            logger.error(f"Error detecting beach regions: {e}")
            return []
    
    def create_roi_polygons(self, image: np.ndarray) -> Dict[str, List[List[int]]]:
        """
        Create ROI polygons for water and beach areas
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary with 'water' and 'beach' polygon coordinates
        """
        try:
            height, width = image.shape[:2]
            
            # Detect regions
            water_contours = self.detect_water_regions(image)
            beach_contours = self.detect_beach_regions(image)
            
            roi_data = {
                'image_width': width,
                'image_height': height,
                'water_polygons': [],
                'beach_polygons': []
            }
            
            # Convert water contours to polygons
            for contour in water_contours:
                # Simplify contour to reduce points
                epsilon = 0.02 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to list of [x, y] coordinates
                polygon = [[int(point[0][0]), int(point[0][1])] for point in simplified]
                roi_data['water_polygons'].append(polygon)
            
            # Convert beach contours to polygons
            for contour in beach_contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                polygon = [[int(point[0][0]), int(point[0][1])] for point in simplified]
                roi_data['beach_polygons'].append(polygon)
            
            return roi_data
            
        except Exception as e:
            logger.error(f"Error creating ROI polygons: {e}")
            return {'image_width': 0, 'image_height': 0, 'water_polygons': [], 'beach_polygons': []}
    
    def save_roi_zones(self, image: np.ndarray, output_path: str = None) -> str:
        """
        Detect and save ROI zones to JSON file
        
        Args:
            image: Input BGR image
            output_path: Optional output path for JSON file
            
        Returns:
            Path to saved JSON file
        """
        try:
            if output_path is None:
                output_path = self.paths_config['roi_zones']
            
            # Create ROI polygons
            roi_data = self.create_roi_polygons(image)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(roi_data, f, indent=2)
            
            logger.info(f"Saved ROI zones to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving ROI zones: {e}")
            raise
    
    def load_roi_zones(self, roi_path: str = None) -> Dict[str, Any]:
        """
        Load ROI zones from JSON file
        
        Args:
            roi_path: Optional path to ROI JSON file
            
        Returns:
            ROI data dictionary
        """
        try:
            if roi_path is None:
                roi_path = self.paths_config['roi_zones']
            
            with open(roi_path, 'r') as f:
                roi_data = json.load(f)
            
            logger.info(f"Loaded ROI zones from: {roi_path}")
            return roi_data
            
        except Exception as e:
            logger.error(f"Error loading ROI zones: {e}")
            return {'water_polygons': [], 'beach_polygons': []}
    
    def classify_point(self, point: Tuple[int, int], roi_data: Dict[str, Any] = None) -> str:
        """
        Classify a point as being in water, beach, or other area
        
        Args:
            point: (x, y) coordinates
            roi_data: Optional ROI data, will load from file if not provided
            
        Returns:
            'water', 'beach', or 'other'
        """
        try:
            if roi_data is None:
                roi_data = self.load_roi_zones()
            
            x, y = point
            
            # Check water polygons
            for polygon in roi_data.get('water_polygons', []):
                if len(polygon) >= 3:  # Valid polygon needs at least 3 points
                    contour = np.array(polygon, dtype=np.int32)
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        return 'water'
            
            # Check beach polygons
            for polygon in roi_data.get('beach_polygons', []):
                if len(polygon) >= 3:
                    contour = np.array(polygon, dtype=np.int32)
                    if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                        return 'beach'
            
            return 'other'
            
        except Exception as e:
            logger.error(f"Error classifying point: {e}")
            return 'other'
    
    def visualize_roi(self, image: np.ndarray, roi_data: Dict[str, Any] = None) -> np.ndarray:
        """
        Visualize ROI zones on the image
        
        Args:
            image: Input BGR image
            roi_data: Optional ROI data
            
        Returns:
            Image with ROI zones drawn
        """
        try:
            if roi_data is None:
                roi_data = self.create_roi_polygons(image)
            
            vis_image = image.copy()
            
            # Draw water polygons in blue
            for polygon in roi_data.get('water_polygons', []):
                if len(polygon) >= 3:
                    contour = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(vis_image, [contour], (255, 100, 100), lineType=cv2.LINE_AA)
                    cv2.polylines(vis_image, [contour], True, (255, 0, 0), 2)
            
            # Draw beach polygons in yellow
            for polygon in roi_data.get('beach_polygons', []):
                if len(polygon) >= 3:
                    contour = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(vis_image, [contour], (100, 255, 255), lineType=cv2.LINE_AA)
                    cv2.polylines(vis_image, [contour], True, (0, 255, 255), 2)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error visualizing ROI: {e}")
            return image

def main():
    """Test ROI detection functionality"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python roi.py <image_path>")
        return
    
    image_path = sys.argv[1]
    detector = ROIDetector()
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Detect and save ROI zones
        roi_path = detector.save_roi_zones(image)
        print(f"ROI zones saved to: {roi_path}")
        
        # Visualize ROI
        vis_image = detector.visualize_roi(image)
        
        # Save visualization
        output_path = "roi_visualization.jpg"
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
        
        # Test point classification
        height, width = image.shape[:2]
        test_points = [
            (width//4, height//2),    # Left side
            (3*width//4, height//2),  # Right side
            (width//2, height//4),    # Top center
            (width//2, 3*height//4)   # Bottom center
        ]
        
        for point in test_points:
            classification = detector.classify_point(point)
            print(f"Point {point}: {classification}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
