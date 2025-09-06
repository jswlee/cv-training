"""
FastAPI tools for the Beach Conditions Agent
"""
import logging
import cv2
import numpy as np
from typing import Dict, Any
from pathlib import Path
import sys
import os

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cv.capture import BeachSnapshotCapture
from src.cv.detection import PeopleDetector
from src.cv.weather import WeatherAnalyzer
from src.cv.roi import ROIDetector

logger = logging.getLogger(__name__)

class BeachAnalysisTools:
    """Collection of tools for beach conditions analysis"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize all analysis tools"""
        self.config_path = config_path
        
        # Initialize components
        self.capture = BeachSnapshotCapture(config_path)
        self.people_detector = PeopleDetector(config_path)
        self.weather_analyzer = WeatherAnalyzer(config_path)
        self.roi_detector = ROIDetector(config_path)
        
        logger.info("Beach analysis tools initialized")
    
    def capture_snapshot(self) -> Dict[str, Any]:
        """
        Capture a current snapshot from the beach livestream
        
        Returns:
            Dictionary with snapshot information or error
        """
        try:
            logger.info("Capturing beach snapshot")
            
            # Capture new snapshot
            snapshot_path = self.capture.capture_current_snapshot()
            
            if not snapshot_path or not Path(snapshot_path).exists():
                return {'error': 'Failed to capture snapshot'}
            
            # Get image info
            image = cv2.imread(snapshot_path)
            if image is None:
                return {'error': 'Failed to load captured image'}
            
            height, width = image.shape[:2]
            file_size = Path(snapshot_path).stat().st_size / (1024 * 1024)  # MB
            
            result = {
                'snapshot_path': snapshot_path,
                'image_width': width,
                'image_height': height,
                'file_size_mb': round(file_size, 2)
            }
            
            logger.info(f"Snapshot captured successfully: {snapshot_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return {'error': str(e)}
    
    def analyze_people(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze people in the beach image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with people analysis results or error
        """
        try:
            logger.info(f"Analyzing people in image: {image_path}")
            
            if not Path(image_path).exists():
                return {'error': f'Image file not found: {image_path}'}
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Failed to load image: {image_path}'}
            
            # Ensure ROI zones exist or create them
            roi_zones_path = self.roi_detector.paths_config['roi_zones']
            if not Path(roi_zones_path).exists():
                logger.info("ROI zones not found, creating them from current image")
                self.roi_detector.save_roi_zones(image)
            
            # Analyze people
            analysis = self.people_detector.analyze_image(image)
            
            result = {
                'total_people': analysis['total_people'],
                'people_in_water': analysis['people_in_water'],
                'people_on_beach': analysis['people_on_beach'],
                'people_other': analysis['people_other'],
                'confidence_stats': analysis['confidence_stats'],
                'detections': analysis['detections']
            }
            
            logger.info(f"People analysis completed: {result['total_people']} people detected")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing people: {e}")
            return {'error': str(e)}
    
    def analyze_weather(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze weather conditions in the beach image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with weather analysis results or error
        """
        try:
            logger.info(f"Analyzing weather in image: {image_path}")
            
            if not Path(image_path).exists():
                return {'error': f'Image file not found: {image_path}'}
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Failed to load image: {image_path}'}
            
            # Analyze weather
            analysis = self.weather_analyzer.analyze_weather(image)
            
            result = {
                'cloud_coverage_percent': analysis['cloud_coverage_percent'],
                'is_raining': analysis['is_raining'],
                'rain_confidence': analysis['rain_confidence'],
                'weather_condition': analysis['weather_condition'],
                'visibility': analysis['visibility'],
                'summary': analysis['summary']
            }
            
            logger.info(f"Weather analysis completed: {result['weather_condition']}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing weather: {e}")
            return {'error': str(e)}
    
    def detect_roi_zones(self, image_path: str) -> Dict[str, Any]:
        """
        Detect and save ROI zones from the beach image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with ROI detection results or error
        """
        try:
            logger.info(f"Detecting ROI zones in image: {image_path}")
            
            if not Path(image_path).exists():
                return {'error': f'Image file not found: {image_path}'}
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Failed to load image: {image_path}'}
            
            # Detect and save ROI zones
            roi_path = self.roi_detector.save_roi_zones(image)
            roi_data = self.roi_detector.load_roi_zones(roi_path)
            
            result = {
                'roi_zones_path': roi_path,
                'image_width': roi_data['image_width'],
                'image_height': roi_data['image_height'],
                'water_polygons': roi_data['water_polygons'],
                'beach_polygons': roi_data['beach_polygons'],
                'zones_detected': len(roi_data['water_polygons']) + len(roi_data['beach_polygons'])
            }
            
            logger.info(f"ROI detection completed: {result['zones_detected']} zones detected")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting ROI zones: {e}")
            return {'error': str(e)}
    
    def get_latest_snapshot(self) -> Dict[str, Any]:
        """
        Get the path to the most recent snapshot
        
        Returns:
            Dictionary with latest snapshot info or error
        """
        try:
            logger.info("Getting latest snapshot")
            
            snapshot_path = self.capture.get_latest_snapshot()
            
            if not snapshot_path or not Path(snapshot_path).exists():
                return {'error': 'No recent snapshots found'}
            
            # Get image info
            image = cv2.imread(snapshot_path)
            if image is None:
                return {'error': 'Failed to load latest snapshot'}
            
            height, width = image.shape[:2]
            file_size = Path(snapshot_path).stat().st_size / (1024 * 1024)  # MB
            
            result = {
                'snapshot_path': snapshot_path,
                'image_width': width,
                'image_height': height,
                'file_size_mb': round(file_size, 2)
            }
            
            logger.info(f"Latest snapshot: {snapshot_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting latest snapshot: {e}")
            return {'error': str(e)}
    
    def analyze_complete_conditions(self, force_new_snapshot: bool = False) -> Dict[str, Any]:
        """
        Perform complete beach conditions analysis
        
        Args:
            force_new_snapshot: Whether to capture a new snapshot
            
        Returns:
            Complete analysis results or error
        """
        try:
            logger.info("Performing complete beach conditions analysis")
            
            # Get or capture snapshot
            if force_new_snapshot:
                snapshot_result = self.capture_snapshot()
            else:
                snapshot_result = self.get_latest_snapshot()
                if 'error' in snapshot_result:
                    # Fallback to capturing new snapshot
                    snapshot_result = self.capture_snapshot()
            
            if 'error' in snapshot_result:
                return {'error': f'Failed to get snapshot: {snapshot_result["error"]}'}
            
            snapshot_path = snapshot_result['snapshot_path']
            
            # Analyze people
            people_result = self.analyze_people(snapshot_path)
            if 'error' in people_result:
                logger.warning(f"People analysis failed: {people_result['error']}")
                people_result = {
                    'total_people': 0,
                    'people_in_water': 0,
                    'people_on_beach': 0,
                    'people_other': 0,
                    'confidence_stats': {'mean': 0, 'min': 0, 'max': 0},
                    'detections': {'water': [], 'beach': [], 'other': []}
                }
            
            # Analyze weather
            weather_result = self.analyze_weather(snapshot_path)
            if 'error' in weather_result:
                logger.warning(f"Weather analysis failed: {weather_result['error']}")
                weather_result = {
                    'cloud_coverage_percent': 0.0,
                    'is_raining': False,
                    'rain_confidence': 0.0,
                    'weather_condition': 'unknown',
                    'visibility': 'unknown',
                    'summary': 'Weather analysis unavailable'
                }
            
            # Combine results
            result = {
                'people': people_result,
                'weather': weather_result,
                'snapshot_info': snapshot_result
            }
            
            logger.info("Complete beach conditions analysis finished")
            return result
            
        except Exception as e:
            logger.error(f"Error in complete conditions analysis: {e}")
            return {'error': str(e)}

def create_tools(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Create and return tool functions for the LangGraph agent
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary of tool functions
    """
    try:
        tools_instance = BeachAnalysisTools(config_path)
        
        # Return tool functions
        return {
            'capture_snapshot': tools_instance.capture_snapshot,
            'analyze_people': tools_instance.analyze_people,
            'analyze_weather': tools_instance.analyze_weather,
            'detect_roi_zones': tools_instance.detect_roi_zones,
            'get_latest_snapshot': tools_instance.get_latest_snapshot,
            'analyze_complete_conditions': tools_instance.analyze_complete_conditions
        }
        
    except Exception as e:
        logger.error(f"Error creating tools: {e}")
        raise
