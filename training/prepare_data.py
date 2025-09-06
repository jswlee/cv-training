"""
Data preparation script for training YOLO people detection model
"""
import os
import shutil
import random
import json
import yaml
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

def prepare_yolo_dataset(
    source_dir: str = "CV_Data/Kaanapali_Beach",
    output_dir: str = "data/training_samples",
    sample_size: int = 500,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    Prepare a subset of images for YOLO training
    
    Args:
        source_dir: Source directory with beach images
        output_dir: Output directory for training data
        sample_size: Number of images to sample for training
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (test = 1 - train - val)
    """
    print(f"Preparing YOLO dataset from {source_dir}")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create YOLO directory structure
    yolo_dir = output_path / "yolo_people"
    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files from source
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(source_path.glob(ext))
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {source_dir}")
    
    print(f"Found {len(image_files)} images in source directory")
    
    # Sample images if we have more than needed
    if len(image_files) > sample_size:
        sampled_files = random.sample(image_files, sample_size)
    else:
        sampled_files = image_files
        print(f"Using all {len(sampled_files)} available images")
    
    # Split into train/val/test
    random.shuffle(sampled_files)
    
    n_train = int(len(sampled_files) * train_ratio)
    n_val = int(len(sampled_files) * val_ratio)
    
    train_files = sampled_files[:n_train]
    val_files = sampled_files[n_train:n_train + n_val]
    test_files = sampled_files[n_train + n_val:]
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Copy images to appropriate directories
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    for split_name, files in splits.items():
        for i, img_file in enumerate(files):
            # Copy image
            dest_img = yolo_dir / "images" / split_name / f"beach_{i:04d}.jpg"
            shutil.copy2(img_file, dest_img)
            
            # Create empty label file (to be filled manually or with annotation tool)
            label_file = yolo_dir / "labels" / split_name / f"beach_{i:04d}.txt"
            label_file.touch()
    
    # Create dataset YAML file
    dataset_yaml = {
        "path": str(yolo_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["person"],
        "nc": 1
    }
    
    with open(yolo_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Dataset prepared in {yolo_dir}")
    print("Next steps:")
    print("1. Annotate images using Label Studio, Roboflow, or similar tool")
    print("2. Export annotations in YOLO format")
    print("3. Run training script: python training/train_yolo.py")
    
    return str(yolo_dir)

def analyze_dataset_statistics(dataset_dir: str):
    """Analyze and print dataset statistics"""
    dataset_path = Path(dataset_dir)
    
    print(f"\nDataset Statistics for {dataset_dir}")
    print("=" * 50)
    
    for split in ["train", "val", "test"]:
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split
        
        if not images_dir.exists():
            continue
            
        image_files = list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))
        
        # Count labeled vs unlabeled
        labeled_count = 0
        total_annotations = 0
        
        for label_file in label_files:
            if label_file.stat().st_size > 0:  # Non-empty label file
                labeled_count += 1
                with open(label_file, 'r') as f:
                    total_annotations += len(f.readlines())
        
        print(f"{split.upper()} SET:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labeled: {labeled_count}")
        print(f"  Unlabeled: {len(image_files) - labeled_count}")
        print(f"  Total annotations: {total_annotations}")
        print()

def create_sample_annotations(dataset_dir: str, num_samples: int = 5):
    """
    Create sample annotation files to demonstrate YOLO format
    This is just for demonstration - real annotations should be done with proper tools
    """
    print(f"Creating {num_samples} sample annotations for demonstration")
    
    dataset_path = Path(dataset_dir)
    train_images = list((dataset_path / "images" / "train").glob("*.jpg"))
    
    if len(train_images) == 0:
        print("No training images found")
        return
    
    # Create sample annotations for first few images
    for i in range(min(num_samples, len(train_images))):
        img_file = train_images[i]
        label_file = dataset_path / "labels" / "train" / f"{img_file.stem}.txt"
        
        # Load image to get dimensions
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        # Create fake annotations (center region of image)
        # Format: class_id center_x center_y width height (normalized)
        sample_annotations = [
            "0 0.5 0.6 0.1 0.2",  # Person in center-bottom
            "0 0.3 0.7 0.08 0.15",  # Person on left
        ]
        
        with open(label_file, 'w') as f:
            f.write('\n'.join(sample_annotations))
    
    print(f"Created sample annotations for {min(num_samples, len(train_images))} images")
    print("Note: These are fake annotations for demonstration only!")
    print("Replace with real annotations from your annotation tool.")

def main():
    """Main function to prepare training data"""
    print("Beach Conditions Agent - Data Preparation")
    print("=" * 50)
    
    # Check if source directory exists
    source_dir = "CV_Data/Kaanapali_Beach"
    if not Path(source_dir).exists():
        print(f"Error: Source directory {source_dir} not found")
        print("Please ensure you have beach images in this directory")
        return
    
    try:
        # Prepare YOLO dataset
        dataset_dir = prepare_yolo_dataset(
            source_dir=source_dir,
            sample_size=300,  # Reasonable size for initial training
            train_ratio=0.7,
            val_ratio=0.2
        )
        
        # Analyze dataset
        analyze_dataset_statistics(dataset_dir)
        
        # Create sample annotations for demonstration
        create_sample_annotations(dataset_dir, num_samples=3)
        
        print("\nData preparation completed successfully!")
        print(f"Dataset location: {dataset_dir}")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
