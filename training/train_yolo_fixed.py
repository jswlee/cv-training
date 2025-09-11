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
    
    # Add comprehensive W&B callbacks for metric tracking
    if use_wandb:
        def on_train_epoch_end(trainer):
            """Log training metrics after each epoch"""
            if not wandb.run:
                return
            
            epoch = trainer.epoch
            metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
            
            # Training losses
            log_dict = {"epoch": epoch}
            
            # Box, class, and DFL losses
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                loss_items = trainer.loss_items
                if len(loss_items) >= 3:
                    log_dict.update({
                        "train/box_loss": float(loss_items[0]),
                        "train/cls_loss": float(loss_items[1]), 
                        "train/dfl_loss": float(loss_items[2]),
                    })
            
            # Learning rates
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                for i, param_group in enumerate(trainer.optimizer.param_groups):
                    log_dict[f"train/lr_pg{i}"] = param_group['lr']
            
            wandb.log(log_dict, step=epoch)
        
        def on_val_end(validator):
            """Log validation metrics after validation"""
            if not wandb.run:
                return
            
            # CORRECTED: Access epoch directly from the validator object
            epoch = validator.epoch
            
            log_dict = {"epoch": epoch}
            
            # CORRECTED: Access validation losses directly from the validator.loss tensor
            if hasattr(validator, 'loss') and validator.loss is not None:
                val_losses = validator.loss
                if len(val_losses) >= 3:
                    log_dict.update({
                        "val/box_loss": float(val_losses[0]),
                        "val/cls_loss": float(val_losses[1]),
                        "val/dfl_loss": float(val_losses[2]),
                    })
            
            # Validation metrics from results
            if hasattr(validator, 'metrics'):
                val_results = validator.metrics
                if hasattr(val_results, 'results_dict'):
                    results_dict = val_results.results_dict
                    
                    # Map YOLO metric names to W&B names
                    metric_mapping = {
                        'metrics/precision(B)': 'val/precision',
                        'metrics/recall(B)': 'val/recall', 
                        'metrics/mAP50(B)': 'val/mAP50',
                        'metrics/mAP50-95(B)': 'val/mAP50-95',
                        'fitness': 'val/fitness'
                    }
                    
                    for yolo_key, wandb_key in metric_mapping.items():
                        if yolo_key in results_dict:
                            log_dict[wandb_key] = float(results_dict[yolo_key])
                
                # Also try direct access to box metrics
                if hasattr(val_results, 'box'):
                    box_metrics = val_results.box
                    if hasattr(box_metrics, 'map'):
                        log_dict['val/mAP50-95'] = float(box_metrics.map)
                    if hasattr(box_metrics, 'map50'):
                        log_dict['val/mAP50'] = float(box_metrics.map50)
                    if hasattr(box_metrics, 'mp'):
                        log_dict['val/precision'] = float(box_metrics.mp)
                    if hasattr(box_metrics, 'mr'):
                        log_dict['val/recall'] = float(box_metrics.mr)
            
            wandb.log(log_dict, step=epoch)
        
        def on_fit_epoch_end(trainer):
            """Log system metrics and other info"""
            if not wandb.run:
                return
                
            epoch = trainer.epoch
            log_dict = {"epoch": epoch}
            
            # GPU memory
            if torch.cuda.is_available():
                log_dict["system/gpu_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
                log_dict["system/gpu_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
            
            # Training speed metrics
            if hasattr(trainer, 'times'):
                times = trainer.times
                if len(times) >= 2:
                    log_dict["system/epoch_time"] = sum(times)
                    log_dict["system/data_time"] = times[0] if len(times) > 0 else 0
                    log_dict["system/forward_time"] = times[1] if len(times) > 1 else 0
            
            wandb.log(log_dict, step=epoch)
        
        # Register callbacks
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_end", on_val_end) 
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