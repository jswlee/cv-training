#!/usr/bin/env python3
"""
Visualize labels inside and outside the ROI mask
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# Import the ROI mask creation function from apply_roi_mask.py
sys.path.append(str(Path(__file__).parent))
from apply_roi_mask import create_roi_mask

def visualize_labels_with_roi(image_path, label_path, output_path, exclude_pool=True):
    """
    Visualize labels and show which ones are inside/outside the ROI
    
    Args:
        image_path: Path to the image
        label_path: Path to the YOLO format label file
        output_path: Path to save the visualization
        exclude_pool: Whether to exclude the pool area
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Create ROI mask
    mask = create_roi_mask(image.shape, exclude_pool)
    
    # Create a visualization image
    vis_image = image.copy()
    
    # Add semi-transparent overlay for masked areas
    overlay = np.zeros_like(vis_image)
    overlay[mask == 0] = [0, 0, 255]  # Red for excluded areas
    alpha = 0.3
    vis_image = cv2.addWeighted(vis_image, 1, overlay, alpha, 0)
    
    # Draw the boundary of the mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)
    
    # Read and draw labels
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls = int(parts[0])
                    # YOLO format: x, y, w, h are normalized (0-1)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # Convert to pixel coordinates
                    x_pixel = int(x_center * width)
                    y_pixel = int(y_center * height)
                    w_pixel = int(w * width)
                    h_pixel = int(h * height)
                    
                    # Calculate box coordinates
                    x1 = int(x_pixel - w_pixel / 2)
                    y1 = int(y_pixel - h_pixel / 2)
                    x2 = int(x_pixel + w_pixel / 2)
                    y2 = int(y_pixel + h_pixel / 2)
                    
                    # Check if center point is in the ROI
                    in_roi = mask[y_pixel, x_pixel] == 1
                    
                    # Draw bounding box with different colors
                    color = (0, 255, 0) if in_roi else (0, 0, 255)  # Green if in ROI, Red if outside
                    thickness = 2
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw center point
                    cv2.circle(vis_image, (x_pixel, y_pixel), 3, color, -1)
                    
                    # Add label text
                    label_text = f"Class {cls}" + (" (IN ROI)" if in_roi else " (OUTSIDE)")
                    cv2.putText(vis_image, label_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error processing label: {e}")
    
    # Save the visualization
    cv2.imwrite(str(output_path), vis_image)
    print(f"Saved visualization to {output_path}")
    return vis_image

def main():
    parser = argparse.ArgumentParser(description="Visualize labels with ROI mask")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image file")
    parser.add_argument("--label", type=str, required=True,
                        help="Path to the YOLO format label file")
    parser.add_argument("--output", type=str, default="label_roi_visualization.jpg",
                        help="Path to save the visualization")
    
    args = parser.parse_args()
    
    visualize_labels_with_roi(args.image, args.label, args.output)

if __name__ == "__main__":
    main()
