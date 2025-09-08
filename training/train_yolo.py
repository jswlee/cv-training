"""
YOLO training script for beach people detection
"""
from pathlib import Path
from ultralytics import YOLO
import torch
import wandb
import yaml

wandb.login(key="573fddab9d08e1d80ff26c4b879932e595d63175")  # Replace with your actual W&B API key

def train_yolo_model(
    dataset_yaml: str = "data/manually_annotated_data/data.yaml",
    model_size: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    output_dir: str = "models",
    config_path: str = "config.yaml"
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
        config_path: Path to config.yaml with W&B settings
    """
    print(f"Training YOLO model for beach detection")
    print(f"Dataset: {dataset_yaml}")
    print(f"Model: {model_size}")
    print(f"Epochs: {epochs}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract W&B settings
    wandb_config = config.get('wandb', {})
    use_wandb = wandb_config.get('enabled', False)
    wandb_project = wandb_config.get('project', 'beach-detection')
    wandb_name = wandb_config.get('name', 'yolo-beach-training')
    
    # Check if dataset exists
    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Initialize W&B if requested
    if use_wandb:
        # Initialize wandb exactly like the example
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "model": model_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "image_size": img_size,
                "device": device
            }
        )
    
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
        name="beach_detection",
        save=True,
        save_period=1,  # Save checkpoint every 1 epochs
        val=True,  # Validates on validation set during training
        plots=True,
        verbose=True
    )
    
    # Save the best model to the specified location
    best_model_path = Path(output_dir) / "beach_detection.pt"
    model.save(str(best_model_path))
    
    print(f"Training completed!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training results: {results}")
    
    if use_wandb:
        wandb.finish()
    
    return str(best_model_path)

def validate_model(model_path: str, dataset_yaml: str = "data/manually_annotated_data/data.yaml"):
    """Validate the trained model on validation set"""
    print(f"Validating model on validation set: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, split='val')  # Explicitly use validation set
    
    print("Validation Results (on validation set):")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    return results

def evaluate_on_test(model_path: str, dataset_yaml: str = "data/manually_annotated_data/data.yaml"):
    """Final evaluation on test set (should be done only once after training)"""
    print(f"Final evaluation on test set: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, split='test')  # Explicitly use test set
    
    print("Test Results (final evaluation on test set):")
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
        test_images_dir = Path("data/manually_annotated_data/test/images")
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
    dataset_yaml = "data/manually_annotated_data/data.yaml"
    if not Path(dataset_yaml).exists():
        print(f"Dataset not found: {dataset_yaml}")
        print("Please ensure your annotated data is in data/manually_annotated_data/")
        return 1
    
    # Check if annotations exist
    labels_dir = Path("data/manually_annotated_data/train/labels")
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        non_empty_labels = [f for f in label_files if f.stat().st_size > 0]
        
        if len(non_empty_labels) == 0:
            print("No annotations found in training set!")
            print("Please ensure your data has annotations")
            return 1
        
        print(f"Found {len(non_empty_labels)} annotated images")
    
    try:
        # Train the model
        model_path = train_yolo_model(
            dataset_yaml=dataset_yaml,
            epochs=30,  # Reasonable number for initial training
            batch_size=4,  # Conservative batch size
            img_size=1920,
            config_path="config.yaml"  # Load W&B settings from config
        )
        
        print("\nValidating trained model on validation set...")
        validate_model(model_path, dataset_yaml)
        
        print("\nPerforming final evaluation on test set...")
        evaluate_on_test(model_path, dataset_yaml)
        
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
