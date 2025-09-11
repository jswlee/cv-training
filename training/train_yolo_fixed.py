#!/usr/bin/env python3
# Fixed YOLO training with optimized hyperparameters and robust W&B logging

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
                "learning_rate": 0.001,
                "warmup_epochs": 5,
                "patience": 50,
            },
        )
        if entity:
            init_kwargs["entity"] = entity
        wandb.init(**init_kwargs)

    # Build model
    model = YOLO(model_size)
    
    # Add comprehensive W&B callbacks for metric tracking
    if use_wandb:
        def on_fit_epoch_end(trainer):
            """
            Logs training and validation metrics at the end of each epoch
            by reading the results.csv file. This is the most robust method.
            """
            if not wandb.run or not hasattr(trainer, 'csv'):
                return

            try:
                # Read the latest row from the results CSV file
                results_df = pd.read_csv(trainer.csv)
                if results_df.empty:
                    return
                
                latest_results = results_df.iloc[-1].to_dict()
                
                # Clean up column names (remove whitespace)
                latest_results = {k.strip(): v for k, v in latest_results.items()}

                # Define a mapping from CSV columns to W&B metric names
                metric_mapping = {
                    'epoch': 'epoch',
                    'train/box_loss': 'train/box_loss',
                    'train/cls_loss': 'train/cls_loss',
                    'train/dfl_loss': 'train/dfl_loss',
                    'metrics/precision(B)': 'val/precision',
                    'metrics/recall(B)': 'val/recall',
                    'metrics/mAP50(B)': 'val/mAP50',
                    'metrics/mAP50-95(B)': 'val/mAP50-95',
                    'val/box_loss': 'val/box_loss',
                    'val/cls_loss': 'val/cls_loss',
                    'val/dfl_loss': 'val/dfl_loss',
                }

                # Prepare the log dictionary
                log_dict = {}
                for csv_key, wandb_key in metric_mapping.items():
                    if csv_key in latest_results:
                        log_dict[wandb_key] = latest_results[csv_key]
                
                # Log learning rates separately
                if hasattr(trainer, 'optimizer'):
                    for i, pg in enumerate(trainer.optimizer.param_groups):
                        log_dict[f'train/lr_pg{i}'] = pg['lr']

                # Log to W&B
                wandb.log(log_dict, step=int(latest_results['epoch']))

            except Exception as e:
                print(f"[W&B Callback WARN] Failed to log metrics: {e}")

        # Register the callback
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

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
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        # Hyperparameters
        lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=5, warmup_momentum=0.8, warmup_bias_lr=0.1,
        box=7.5, cls=0.5, dfl=1.5,
        # Data augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,
        # Validation settings
        patience=50, close_mosaic=10,
    )

    # Determine run dir and best weights
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
        csv_path = run_dir / "results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Log the full results table
                wandb.log({"training/results_table": wandb.Table(dataframe=df)})
                
                # --- LOG SUMMARY METRICS (as requested from docs) ---
                # Find the epoch with the best mAP50-95
                best_epoch = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
                
                # Update wandb.summary with the best metrics
                wandb.summary["best_epoch"] = int(best_epoch['epoch'])
                wandb.summary["best_mAP50-95"] = best_epoch['metrics/mAP50-95(B)']
                wandb.summary["best_mAP50"] = best_epoch['metrics/mAP50(B)']
                wandb.summary["best_precision"] = best_epoch['metrics/precision(B)']
                wandb.summary["best_recall"] = best_epoch['metrics/recall(B)']

            except Exception as e:
                print(f"[warn] Could not parse results.csv for summary: {e}")

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
            model_size="yolov8x.pt",
            epochs=150,
            batch_size=1,
            img_size=1920,
            run_name="beach_detection_fixed",
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