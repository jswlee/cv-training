"""
YOLO training script for beach people detection
"""
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

def train_yolo_model(
    dataset_yaml: str = "data/training_samples/yolo_people/dataset.yaml",
    model_size: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    output_dir: str = "models"
):
    """
    Train YOLO model for people detection on beach images
    
    Args:
        dataset_yaml: Path to dataset YAML file
        model_size: YOLO model size (yolov8n.pt, yolov8s.pt, etc.)
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        output_dir: Output directory for trained model
    """
    print(f"Training YOLO model for beach people detection")
    print(f"Dataset: {dataset_yaml}")
    print(f"Model: {model_size}")
    print(f"Epochs: {epochs}")
    
    # Check if dataset exists
    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLO model
    model = YOLO(model_size)
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name="yolo_people_training",
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        val=True,
        plots=True,
        verbose=True
    )
    
    # Save the best model to the specified location
    best_model_path = Path(output_dir) / "yolo_people.pt"
    model.save(str(best_model_path))
    
    print(f"Training completed!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training results: {results}")
    
    return str(best_model_path)

def validate_model(model_path: str, dataset_yaml: str):
    """Validate the trained model"""
    print(f"Validating model: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml)
    
    print("Validation Results:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    return results

def test_model_inference(model_path: str, test_image: str = None):
    """Test model inference on a sample image"""
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    
    # Find a test image if not provided
    if test_image is None:
        test_images_dir = Path("data/training_samples/yolo_people/images/test")
        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*.jpg"))
            if test_images:
                test_image = str(test_images[0])
    
    if not test_image or not Path(test_image).exists():
        print("No test image available for inference test")
        return
    
    print(f"Testing inference on: {test_image}")
    
    model = YOLO(model_path)
    results = model(test_image)
    
    # Print detection results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"Detected {len(boxes)} objects")
            for box in boxes:
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                print(f"  Class: {cls}, Confidence: {conf:.3f}")
        else:
            print("No objects detected")
    
    # Save result image
    output_path = "inference_test_result.jpg"
    results[0].save(output_path)
    print(f"Result saved to: {output_path}")

def main():
    """Main training function"""
    print("Beach Conditions Agent - YOLO Training")
    print("=" * 50)
    
    # Check if dataset is prepared
    dataset_yaml = "data/training_samples/yolo_people/dataset.yaml"
    if not Path(dataset_yaml).exists():
        print(f"Dataset not found: {dataset_yaml}")
        print("Please run 'python training/prepare_data.py' first")
        return 1
    
    # Check if annotations exist
    labels_dir = Path("data/training_samples/yolo_people/labels/train")
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        non_empty_labels = [f for f in label_files if f.stat().st_size > 0]
        
        if len(non_empty_labels) == 0:
            print("No annotations found in training set!")
            print("Please annotate your images before training:")
            print("1. Use Label Studio, Roboflow, or similar annotation tool")
            print("2. Export annotations in YOLO format")
            print("3. Place label files in data/training_samples/yolo_people/labels/")
            return 1
        
        print(f"Found {len(non_empty_labels)} annotated images")
    
    try:
        # Train the model
        model_path = train_yolo_model(
            dataset_yaml=dataset_yaml,
            epochs=30,  # Reasonable number for initial training
            batch_size=8,  # Conservative batch size
            img_size=640
        )
        
        print("\nValidating trained model...")
        validate_model(model_path, dataset_yaml)
        
        print("\nTesting inference...")
        test_model_inference(model_path)
        
        print(f"\nTraining completed successfully!")
        print(f"Trained model: {model_path}")
        print("You can now use this model with the beach conditions agent.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
