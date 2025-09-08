"""
Apply ROI mask to exclude pool area from training images
"""
import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
import shutil

def create_roi_mask(image_shape, exclude_pool=True):
    """
    Create a binary mask for the ROI
    
    Args:
        image_shape: Shape of the image (height, width)
        exclude_pool: Whether to exclude the pool area (bottom right)
        
    Returns:
        Binary mask where 1 indicates the ROI
    """
    height, width = image_shape[:2]
    
    # Create a white mask (all 1s)
    mask = np.ones((height, width), dtype=np.uint8)
    
    if exclude_pool:
        # Define the pool area as a polygon in the bottom right
        # Adjusted based on the sample image
        pool_vertices = np.array([
            [width * 0.11, height],     # Bottom left of pool area
            [width, height],           # Bottom right
            [width, height * 0.60],     # Middle right
            [width * 0.71, height * 0.73]  # Upper left corner of pool area
        ], dtype=np.int32)
        
        # Fill the pool area with black (0s)
        cv2.fillPoly(mask, [pool_vertices], 0)
    
    return mask

def apply_mask_to_image(image, mask):
    """
    Apply the binary mask to an image
    
    Args:
        image: Input image
        mask: Binary mask
        
    Returns:
        Masked image
    """
    # Convert mask to 3 channels if the image is color
    if len(image.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply the mask
    masked_image = image * mask
    
    return masked_image

def process_dataset(input_dir, output_dir, exclude_pool=True):
    """
    Process all images in a dataset directory
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        exclude_pool: Whether to exclude the pool area
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for img_path in image_files:
        # Read the image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not read {img_path}")
            continue
        
        # Create mask for this image
        mask = create_roi_mask(image.shape, exclude_pool)
        
        # Apply mask to image
        masked_image = apply_mask_to_image(image, mask)
        
        # Save the masked image
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), masked_image)
        
    print(f"Processed {len(image_files)} images to {output_dir}")

def process_yolo_dataset(dataset_dir, output_dir):
    """
    Process a YOLO dataset with train/valid/test splits
    
    Args:
        dataset_dir: Input directory containing YOLO dataset
        output_dir: Output directory for processed dataset
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_images_dir = dataset_path / split / 'images'
        split_labels_dir = dataset_path / split / 'labels'
        
        if not split_images_dir.exists():
            print(f"Split directory not found: {split_images_dir}")
            continue
        
        # Create output directories for this split
        output_split_images_dir = output_path / split / 'images'
        output_split_labels_dir = output_path / split / 'labels'
        
        output_split_images_dir.mkdir(parents=True, exist_ok=True)
        output_split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        process_dataset(split_images_dir, output_split_images_dir)
        
        # Copy label files (no changes needed for labels)
        if split_labels_dir.exists():
            for label_file in split_labels_dir.glob('*.txt'):
                shutil.copy(label_file, output_split_labels_dir / label_file.name)
    
    # Copy the data.yaml file if it exists
    yaml_file = dataset_path / 'data.yaml'
    if yaml_file.exists():
        with open(yaml_file, 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        # Update paths in the YAML file
        data_yaml['path'] = str(output_path.absolute())
        data_yaml['train'] = str(output_path / 'train' / 'images')
        data_yaml['val'] = str(output_path / 'valid' / 'images')
        data_yaml['test'] = str(output_path / 'test' / 'images')
        
        # Write the updated YAML file
        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Processed YOLO dataset from {dataset_dir} to {output_dir}")
    return str(output_path / 'data.yaml')

def visualize_roi(image_path, output_path, exclude_pool=True):
    """
    Visualize the ROI on an image
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the visualization
        exclude_pool: Whether to exclude the pool area
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return
    
    # Create mask
    mask = create_roi_mask(image.shape, exclude_pool)
    
    # Create a visualization with the ROI outlined in red
    vis_image = image.copy()
    
    # Create a colored overlay for the mask
    overlay = np.zeros_like(vis_image)
    overlay[mask == 0] = [0, 0, 255]  # Red for excluded areas
    
    # Blend the overlay with the original image
    alpha = 0.3
    vis_image = cv2.addWeighted(vis_image, 1, overlay, alpha, 0)
    
    # Draw the boundary of the mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)
    
    # Save the visualization
    cv2.imwrite(output_path, vis_image)
    print(f"Saved ROI visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Apply ROI mask to exclude pool area from images")
    parser.add_argument("--input", type=str, default="data/manually_annotated_data",
                        help="Input directory containing the YOLO dataset")
    parser.add_argument("--output", type=str, default="data/roi_masked_data",
                        help="Output directory for the processed dataset")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the ROI on a sample image")
    parser.add_argument("--sample-image", type=str,
                        help="Sample image for visualization")
    
    args = parser.parse_args()
    
    if args.visualize and args.sample_image:
        visualize_roi(args.sample_image, "roi_visualization.jpg")
    else:
        # Process the entire dataset
        new_yaml_path = process_yolo_dataset(args.input, args.output)
        print(f"New dataset YAML file: {new_yaml_path}")
        print(f"Use this YAML file for training: {new_yaml_path}")

if __name__ == "__main__":
    main()
