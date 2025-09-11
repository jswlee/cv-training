#!/usr/bin/env python3
# Fixed YOLO training with optimized hyperparameters for fine-tuning

from pathlib import Path
import yaml
import pandas as pd
import torch
import wandb
import os
import sys
from ultralytics import YOLO

# Volume configuration
VOLUME_PATH = Path("/media/volume/yolo-training-data")

def check_volume_available():
    """Check if the attached volume is available and writable"""
    if not VOLUME_PATH.exists():
        raise RuntimeError(
            f"Attached volume not found at {VOLUME_PATH}\n"
            "Please ensure the volume is mounted and accessible."
        )
    
    if not os.access(VOLUME_PATH, os.W_OK):
        raise RuntimeError(
            f"Volume at {VOLUME_PATH} is not writable\n"
            "Please check volume permissions."
        )
    
    print(f"✅ Volume available at: {VOLUME_PATH}")
    return True

def train_yolo_model(
    dataset_yaml: str = "data/roi_filtered_data/data.yaml",
    model_size: str = "yolov8x.pt",
    epochs: int = 150,
    batch_size: int = 4,
    img_size: int = 1920,
    output_dir: str = None,  # Will be set to volume path
    run_name: str = "beach_detection_fixed",
    config_path: str = "config.yaml",
):
    print("Beach Conditions Agent - YOLO Training (Fixed)")
    print("=" * 50)

    # Check volume availability first
    check_volume_available()
    
    # Set output directory to volume path
    if output_dir is None:
        output_dir = str(VOLUME_PATH / "runs")
    
    # Create volume subdirectories
    (VOLUME_PATH / "runs").mkdir(parents=True, exist_ok=True)
    (VOLUME_PATH / "models").mkdir(parents=True, exist_ok=True)
    (VOLUME_PATH / "wandb").mkdir(parents=True, exist_ok=True)

    dspath = Path(dataset_yaml)
    if not dspath.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    # W&B config
    wandb_cfg = {}
    if Path(config_path).exists():
        cfg = yaml.safe_load(Path(config_path).read_text())
        wandb_cfg = cfg.get("wandb", {})

    use_wandb = bool(wandb_cfg.get("enabled", True))
    project = wandb_cfg.get("project", "beach-detection")
    run_display = wandb_cfg.get("name", run_name)
    entity = wandb_cfg.get("entity")

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Init W&B with volume path
    if use_wandb:
        # Set W&B directory to volume
        os.environ["WANDB_DIR"] = str(VOLUME_PATH / "wandb")
        
        init_kwargs = dict(
            project=project,
            name=run_display,
            config={
                "model": model_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "image_size": img_size,
                "device": device,
                "dataset": dataset_yaml,
                "learning_rate": 0.001,  # Lower LR for fine-tuning
                "warmup_epochs": 5,
                "patience": 50,
            },
        )
        if entity:
            init_kwargs["entity"] = entity
        wandb.init(**init_kwargs)

    # Build model
    model = YOLO(model_size)

    # Train with optimized hyperparameters for fine-tuning
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name=run_name,
        save=True,
        save_period=10,  # Save less frequently
        val=True,
        plots=True,
        verbose=True,
        # Optimized hyperparameters for fine-tuning
        lr0=0.001,          # Lower initial learning rate
        lrf=0.01,           # Lower final learning rate
        momentum=0.937,     # Standard momentum
        weight_decay=0.0005, # Standard weight decay
        warmup_epochs=5,    # Longer warmup for stability
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,            # Box loss gain
        cls=0.5,            # Class loss gain  
        dfl=1.5,            # DFL loss gain
        pose=12.0,          # Pose loss gain
        kobj=1.0,           # Keypoint obj loss gain
        label_smoothing=0.0, # No label smoothing initially
        nbs=64,             # Nominal batch size
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,        # No dropout initially
        # Data augmentation (reduced for fine-tuning)
        hsv_h=0.015,        # Hue augmentation
        hsv_s=0.7,          # Saturation augmentation  
        hsv_v=0.4,          # Value augmentation
        degrees=0.0,        # No rotation initially
        translate=0.1,      # Small translation
        scale=0.5,          # Scale augmentation
        shear=0.0,          # No shear initially
        perspective=0.0,    # No perspective initially
        flipud=0.0,         # No vertical flip
        fliplr=0.5,         # Horizontal flip
        mosaic=1.0,         # Use mosaic augmentation
        mixup=0.0,          # No mixup initially
        copy_paste=0.0,     # No copy-paste initially
        # Validation settings
        patience=50,        # Early stopping patience
        close_mosaic=10,    # Disable mosaic in last N epochs
    )

    # Determine run dir and best weights
    run_dir = Path(output_dir) / run_name
    if hasattr(results, 'save_dir') and results.save_dir:
        run_dir = Path(results.save_dir)
    
    best_auto = run_dir / "weights" / "best.pt"
    best_model_path = VOLUME_PATH / "models" / f"{run_name}.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if best_auto.exists():
        best_model_path.write_bytes(best_auto.read_bytes())
    else:
        model.save(str(best_model_path))

    print(f"Training complete. Best model saved to: {best_model_path}")

    if use_wandb:
        # Log final results
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                wandb.log({"training/results_table": wandb.Table(dataframe=df)})
                if len(df):
                    last = df.iloc[-1].to_dict()
                    wandb.log({f"summary/{k}": v for k, v in last.items()})
            except Exception as e:
                print(f"[warn] Could not parse results.csv: {e}")

        # Log plots
        for name in ["results.png", "PR_curve.png", "F1_curve.png", "confusion_matrix.png"]:
            p = run_dir / name
            if p.exists():
                wandb.log({f"plots/{p.stem}": wandb.Image(str(p))})

        # Upload model artifact
        if best_model_path.exists():
            artifact = wandb.Artifact(name=f"beach-people-model-{wandb.run.id}", type="model")
            artifact.add_file(str(best_model_path))
            wandb.log_artifact(artifact)
        
        wandb.finish()

    return str(best_model_path)

def main():
    dataset_yaml = "data/roi_filtered_data/data.yaml"
    
    try:
        model_path = train_yolo_model(
            dataset_yaml=dataset_yaml,
            model_size="yolov8x.pt",  # Using YOLOv8x as requested
            epochs=150,
            batch_size=1,  # Reduced for YOLOv8x memory requirements
            img_size=1920,
            output_dir=None,  # Will use volume path
            run_name="beach_detection_fixed",
            config_path="config.yaml",
        )
        print(f"\nTraining completed successfully!")
        print(f"Trained model: {model_path}")
        
        # Quick validation
        model = YOLO(model_path)
        results = model.val(data=dataset_yaml)
        print(f"Final mAP50: {results.box.map50:.3f}")
        print(f"Final mAP50-95: {results.box.map:.3f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
