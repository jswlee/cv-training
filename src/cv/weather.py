"""
Weather analysis using computer vision techniques
Detects cloud coverage and rain conditions from beach images
"""
import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple
import yaml

logger = logging.getLogger(__name__)

class WeatherAnalyzer:
    """Computer vision-based weather analysis for beach conditions"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize weather analyzer with configuration"""
        self.config = self._load_config(config_path)
        self.weather_config = self.config['weather']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def detect_sky_region(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the sky region in the image
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where sky pixels are white (255)
        """
        try:
            height, width = image.shape[:2]
            
            # Simple approach: assume top portion of image is sky
            sky_region_top = self.weather_config['sky_region_top']
            sky_height = int(height * sky_region_top)
            
            # Create basic sky mask (top portion of image)
            sky_mask = np.zeros((height, width), dtype=np.uint8)
            sky_mask[:sky_height, :] = 255
            
            # Refine sky detection using color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Sky is typically blue or gray/white
            # Blue sky range
            blue_lower = np.array([100, 30, 30])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            # Gray/white sky range (cloudy)
            gray_lower = np.array([0, 0, 150])
            gray_upper = np.array([180, 30, 255])
            gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
            
            # Combine color-based sky detection
            color_sky_mask = cv2.bitwise_or(blue_mask, gray_mask)
            
            # Combine with region-based detection
            refined_sky_mask = cv2.bitwise_and(sky_mask, color_sky_mask)
            
            # Apply morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            refined_sky_mask = cv2.morphologyEx(refined_sky_mask, cv2.MORPH_CLOSE, kernel)
            
            return refined_sky_mask
            
        except Exception as e:
            logger.error(f"Error detecting sky region: {e}")
            # Return simple top region mask as fallback
            height, width = image.shape[:2]
            sky_height = int(height * 0.4)
            fallback_mask = np.zeros((height, width), dtype=np.uint8)
            fallback_mask[:sky_height, :] = 255
            return fallback_mask
    
    def estimate_cloud_coverage(self, image: np.ndarray) -> float:
        """
        Estimate cloud coverage percentage
        
        Args:
            image: Input BGR image
            
        Returns:
            Cloud coverage percentage (0-100)
        """
        try:
            # Get sky region
            sky_mask = self.detect_sky_region(image)
            
            if np.sum(sky_mask) == 0:
                logger.warning("No sky region detected")
                return 0.0
            
            # Convert to HSV for better cloud detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Clouds are typically bright in the value channel
            cloud_threshold = self.weather_config['cloud_threshold']
            cloud_mask = hsv[:, :, 2] > cloud_threshold
            
            # Apply sky mask to focus only on sky region
            sky_cloud_mask = cv2.bitwise_and(cloud_mask.astype(np.uint8) * 255, sky_mask)
            
            # Calculate cloud coverage percentage
            total_sky_pixels = np.sum(sky_mask > 0)
            cloud_pixels = np.sum(sky_cloud_mask > 0)
            
            if total_sky_pixels == 0:
                return 0.0
            
            coverage_percentage = (cloud_pixels / total_sky_pixels) * 100
            coverage_percentage = min(100.0, max(0.0, coverage_percentage))
            
            logger.info(f"Estimated cloud coverage: {coverage_percentage:.1f}%")
            return coverage_percentage
            
        except Exception as e:
            logger.error(f"Error estimating cloud coverage: {e}")
            return 0.0
    
    def detect_rain(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect rain in the image using streak detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (is_raining, confidence_score)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Edge detection to find streaks
            edges = cv2.Canny(blurred, 50, 150)
            
            # Create vertical kernel to detect rain streaks
            vertical_kernel = np.ones((15, 1), np.uint8)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Calculate rain score based on vertical line density
            total_pixels = image.shape[0] * image.shape[1]
            rain_pixels = np.sum(vertical_lines > 0)
            rain_score = rain_pixels / total_pixels
            
            # Additional checks for rain characteristics
            # Rain often reduces overall brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Normalize brightness and contrast factors
            brightness_factor = max(0, (150 - brightness) / 150)  # Lower brightness suggests rain
            contrast_factor = max(0, (50 - contrast) / 50)        # Lower contrast suggests rain
            
            # Combine factors
            combined_score = rain_score + (brightness_factor * 0.1) + (contrast_factor * 0.1)
            
            # Determine if it's raining based on threshold
            rain_threshold = self.weather_config['rain_streak_threshold']
            is_raining = combined_score > rain_threshold
            
            confidence = min(1.0, combined_score / rain_threshold)
            
            logger.info(f"Rain detection - Score: {combined_score:.4f}, Raining: {is_raining}, Confidence: {confidence:.2f}")
            
            return is_raining, confidence
            
        except Exception as e:
            logger.error(f"Error detecting rain: {e}")
            return False, 0.0
    
    def analyze_weather(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Complete weather analysis of the image
        
        Args:
            image: Input BGR image
            
        Returns:
            Weather analysis results
        """
        try:
            # Estimate cloud coverage
            cloud_coverage = self.estimate_cloud_coverage(image)
            
            # Detect rain
            is_raining, rain_confidence = self.detect_rain(image)
            
            # Determine overall weather condition
            if is_raining:
                condition = "rainy"
            elif cloud_coverage > 80:
                condition = "overcast"
            elif cloud_coverage > 50:
                condition = "cloudy"
            elif cloud_coverage > 20:
                condition = "partly_cloudy"
            else:
                condition = "clear"
            
            # Create weather summary
            analysis = {
                'cloud_coverage_percent': round(cloud_coverage, 1),
                'is_raining': is_raining,
                'rain_confidence': round(rain_confidence, 2),
                'weather_condition': condition,
                'visibility': self._estimate_visibility(image),
                'summary': self._generate_weather_summary(cloud_coverage, is_raining, condition)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing weather: {e}")
            return {
                'cloud_coverage_percent': 0.0,
                'is_raining': False,
                'rain_confidence': 0.0,
                'weather_condition': 'unknown',
                'visibility': 'good',
                'summary': 'Weather analysis unavailable'
            }
    
    def _estimate_visibility(self, image: np.ndarray) -> str:
        """Estimate visibility based on image characteristics"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate contrast and brightness
            contrast = np.std(gray)
            brightness = np.mean(gray)
            
            # Simple visibility estimation
            if contrast > 60 and brightness > 100:
                return "excellent"
            elif contrast > 40 and brightness > 80:
                return "good"
            elif contrast > 25 and brightness > 60:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error estimating visibility: {e}")
            return "unknown"
    
    def _generate_weather_summary(self, cloud_coverage: float, is_raining: bool, condition: str) -> str:
        """Generate human-readable weather summary"""
        try:
            if is_raining:
                return f"Rainy conditions with {cloud_coverage:.0f}% cloud cover"
            elif condition == "clear":
                return f"Clear skies with {cloud_coverage:.0f}% cloud cover"
            elif condition == "partly_cloudy":
                return f"Partly cloudy with {cloud_coverage:.0f}% cloud cover"
            elif condition == "cloudy":
                return f"Cloudy conditions with {cloud_coverage:.0f}% cloud cover"
            elif condition == "overcast":
                return f"Overcast skies with {cloud_coverage:.0f}% cloud cover"
            else:
                return f"Weather conditions unclear"
                
        except Exception as e:
            logger.error(f"Error generating weather summary: {e}")
            return "Weather summary unavailable"
    
    def visualize_weather_analysis(self, image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Visualize weather analysis on the image
        
        Args:
            image: Input BGR image
            analysis: Weather analysis results
            
        Returns:
            Image with weather information overlaid
        """
        try:
            vis_image = image.copy()
            
            # Add weather information text
            info_lines = [
                f"Weather: {analysis['weather_condition'].replace('_', ' ').title()}",
                f"Cloud Coverage: {analysis['cloud_coverage_percent']}%",
                f"Rain: {'Yes' if analysis['is_raining'] else 'No'} ({analysis['rain_confidence']:.2f})",
                f"Visibility: {analysis['visibility'].title()}"
            ]
            
            # Draw background rectangle for text
            text_height = 25 * len(info_lines) + 20
            cv2.rectangle(vis_image, (10, 10), (400, text_height), (0, 0, 0), -1)
            cv2.rectangle(vis_image, (10, 10), (400, text_height), (255, 255, 255), 2)
            
            # Draw text
            for i, line in enumerate(info_lines):
                y_pos = 35 + i * 25
                cv2.putText(vis_image, line, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Visualize sky region
            sky_mask = self.detect_sky_region(image)
            sky_overlay = cv2.bitwise_and(vis_image, vis_image, mask=sky_mask)
            sky_overlay = cv2.addWeighted(sky_overlay, 0.3, vis_image, 0.7, 0)
            
            return sky_overlay
            
        except Exception as e:
            logger.error(f"Error visualizing weather analysis: {e}")
            return image

def main():
    """Test weather analysis functionality"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python weather.py <image_path>")
        return
    
    image_path = sys.argv[1]
    analyzer = WeatherAnalyzer()
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Analyze weather
        analysis = analyzer.analyze_weather(image)
        
        # Print results
        print(f"Weather Condition: {analysis['weather_condition']}")
        print(f"Cloud Coverage: {analysis['cloud_coverage_percent']}%")
        print(f"Is Raining: {analysis['is_raining']} (confidence: {analysis['rain_confidence']})")
        print(f"Visibility: {analysis['visibility']}")
        print(f"Summary: {analysis['summary']}")
        
        # Visualize results
        vis_image = analyzer.visualize_weather_analysis(image, analysis)
        
        # Save visualization
        output_path = "weather_analysis_result.jpg"
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
