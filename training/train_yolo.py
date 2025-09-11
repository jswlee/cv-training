#!/usr/bin/env python3
# YOLO training with robust W&B logging (callbacks registered via model.add_callback)

from pathlib import Path
import yaml
import pandas as pd
import torch
import wandb
import os
import argparse
from ultralytics import YOLO


# ----------------------------- helpers -----------------------------------------

def summarize_dataset(dataset_yaml_path: str) -> dict:
    cfg = yaml.safe_load(Path(dataset_yaml_path).read_text())

    def _count_images(p):
        if not p:
            return 0
        p = Path(p)
        if not p.exists():
            return 0
        if p.is_file() and p.suffix.lower() in {".txt", ".csv"}:
            try:
                return sum(1 for _ in open(p))
            except Exception:
                return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sum(1 for f in p.rglob("*") if f.suffix.lower() in exts)

    names = cfg.get("names")
    if isinstance(names, dict):
        cls_list = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        cls_list = names
    else:
        cls_list = []

    return {
        "dataset_yaml": dataset_yaml_path,
        "train_images": _count_images(cfg.get("train")),
        "val_images": _count_images(cfg.get("val")),
        "test_images": _count_images(cfg.get("test")),
        "nc": len(cls_list) if cls_list else None,
        "names": ", ".join(cls_list) if cls_list else None,
    }


def find_run_dir(results, default_dir: Path) -> Path:
    if hasattr(results, "save_dir") and results.save_dir:
        return Path(results.save_dir)
    return default_dir


def log_ultralytics_outputs_to_wandb(run_dir: Path, best_model_path: Path):
    if not wandb.run:
        return

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

    for name in [
        "results.png",
        "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png",
        "confusion_matrix.png",
    ]:
        p = run_dir / name
        if p.exists():
            wandb.log({f"plots/{p.stem}": wandb.Image(str(p))})

    if best_model_path.exists():
        artifact = wandb.Artifact(name=f"beach-people-model-{wandb.run.id}", type="model")
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)


# ----------------------------- training ----------------------------------------

def get_training_paths(config_path: str = "config.yaml"):
    """Get training output paths from config, with fallback to local paths."""
    if Path(config_path).exists():
        cfg = yaml.safe_load(Path(config_path).read_text())
        training_outputs = cfg.get("training_outputs", {})
        base_path = training_outputs.get("base_path")
        
        if base_path and base_path != "/media/volume/yolo-training-data":
            return {
                "models_dir": Path(base_path) / training_outputs.get("models_dir", "models"),
                "runs_dir": Path(base_path) / training_outputs.get("runs_dir", "runs"),
                "wandb_dir": Path(base_path) / training_outputs.get("wandb_dir", "wandb")
            }
    
    # Fallback to local paths
    return {
        "models_dir": Path("models"),
        "runs_dir": Path("runs"),
        "wandb_dir": Path("wandb")
    }


def train_yolo_model(
    dataset_yaml: str = "data/roi_filtered_data/data.yaml",
    model_size: str = "yolov8n.pt",
    epochs: int = 30,
    batch_size: int = 2,
    img_size: int = 1920,
    output_dir: str = None,
    run_name: str = "beach_detection",
    config_path: str = "config.yaml",
):
    print("Beach Conditions Agent - YOLO Training")
    print("=" * 50)

    dspath = Path(dataset_yaml)
    if not dspath.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            "Please create the ROI-masked dataset first."
        )

    # Get training paths from config
    training_paths = get_training_paths(config_path)
    
    # Use configured output directory or fallback
    if output_dir is None:
        output_dir = str(training_paths["runs_dir"])
    
    # Create output directories
    for path in training_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Set W&B directory if configured
    if "wandb_dir" in training_paths:
        os.environ["WANDB_DIR"] = str(training_paths["wandb_dir"])
    
    print(f"Training outputs will be saved to: {output_dir}")
    print(f"Models will be saved to: {training_paths['models_dir']}")
    print(f"W&B logs will be saved to: {training_paths['wandb_dir']}")

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

    # Init W&B before training so Ultralytics attaches its logger
    if use_wandb:
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
            },
        )
        if entity:
            init_kwargs["entity"] = entity
        wandb.init(**init_kwargs)

        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        # Dataset summary (not time-series)
        ds = summarize_dataset(dataset_yaml)
        for k, v in ds.items():
            wandb.run.summary[f"dataset/{k}"] = v

    # Build model
    model = YOLO(model_size)

    # Optional: watch gradients/params
    if use_wandb and hasattr(model, "model"):
        wandb.watch(model.model, log="gradients", log_freq=100)

    # ---- register callbacks (NOT via train(..., callbacks=...)) ----
    def on_val_end(trainer):
        if not wandb.run:
            return
        sd = Path(trainer.save_dir)
        imgs = sorted(sd.glob("val_batch*.jpg"))[:6]
        if imgs:
            wandb.log(
                {"samples/val_batch": [wandb.Image(str(p)) for p in imgs],
                 "epoch": trainer.epoch}
            )

    def on_fit_epoch_end(trainer):
        if not wandb.run:
            return
        gpu_mem = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
        wandb.log({"system/gpu_mem_gb": gpu_mem, "epoch": trainer.epoch})

    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # Train (Ultralytics streams scalars/plots to W&B automatically)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name=run_name,
        save=True,
        save_period=1,
        val=True,
        plots=True,
        verbose=True,
    )

    # Determine run dir and best weights
    run_dir = find_run_dir(results, Path(output_dir) / run_name)
    best_auto = run_dir / "weights" / "best.pt"
    best_model_path = training_paths["models_dir"] / f"{run_name}.pt"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    if best_auto.exists():
        best_model_path.write_bytes(best_auto.read_bytes())
    else:
        model.save(str(best_model_path))

    print(f"Training complete. Best model saved to: {best_model_path}")

    if use_wandb:
        log_ultralytics_outputs_to_wandb(run_dir, best_model_path)
        wandb.finish()

    return str(best_model_path)


# ----------------------------- eval / inference -------------------------------

def validate_model(model_path: str, dataset_yaml: str = "data/roi_filtered_data/data.yaml"):
    print(f"Validating on validation set: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, split="val")
    print("Validation results:", getattr(results, "results_dict", results))
    return results


def evaluate_on_test(model_path: str, dataset_yaml: str = "data/roi_filtered_data/data.yaml"):
    print(f"Final evaluation on test set: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml, split="test")
    print("Test results:", getattr(results, "results_dict", results))
    return results


def test_model_inference(model_path: str, test_image: str = None, config_path: str = "config.yaml"):
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    if test_image is None:
        test_dir = Path("data/roi_masked_data/test/images")
        if test_dir.exists():
            imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if imgs:
                test_image = str(imgs[0])
    if not test_image or not Path(test_image).exists():
        print("No test image available for inference test")
        return
    
    # Get training paths from config
    training_paths = get_training_paths(config_path)
    
    print(f"Testing inference on: {test_image}")
    model = YOLO(model_path)
    results = model(test_image)
    
    # Save to external volume
    out = training_paths["models_dir"] / "inference_test_result.jpg"
    results[0].save(out)
    print(f"Result saved to: {out}")
    
    # Create a symlink in the project directory for convenience
    local_out = Path("inference_test_result.jpg")
    if local_out.exists():
        local_out.unlink()
    try:
        local_out.symlink_to(out)
        print(f"Created symlink at {local_out}")
    except Exception as e:
        print(f"Could not create symlink: {e}")
        # Fall back to copy if symlink fails
        try:
            import shutil
            shutil.copy(out, local_out)
            print(f"Copied result to {local_out}")
        except Exception:
            pass


# --------------------------------- main ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument("--model-size", default="yolov8n.pt", help="YOLO model size")
    args = parser.parse_args()
    
    dataset_yaml = "data/roi_filtered_data/data.yaml"

    labels_dir = Path("data/roi_filtered_data/train/labels")
    if labels_dir.exists():
        non_empty = [f for f in labels_dir.glob("*.txt") if f.stat().st_size > 0]
        if not non_empty:
            print("No annotations found in training set!")
            return 1

    try:
        model_path = train_yolo_model(
            dataset_yaml=dataset_yaml,
            epochs=200,
            batch_size=1,
            img_size=1920,
            run_name="beach_detection",
            config_path="config.yaml",
            model_size=args.model_size,
        )
        print("\nValidating trained model...")
        validate_model(model_path, dataset_yaml)
        print("\nEvaluating on test set...")
        evaluate_on_test(model_path, dataset_yaml)
        print("\nTesting single-image inference...")
        test_model_inference(model_path, config_path="config.yaml")
        print("\nDone.")
        print(f"Trained model: {model_path}")
    except Exception as e:
        print(f"Training failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
