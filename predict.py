#!/usr/bin/env python3
"""
Simple YOLO prediction script.

Usage examples:
  python predict.py \
    --model "/Users/jlee/Desktop/github/cv-training/models/beach_detection_100e30p20we1e-5lr10f.pt" \
    --source data/roi_filtered_data/test/images \
    --imgsz 1280 --conf 0.25 --device 0 --save

  python predict.py --model models/beach_detection_100e30p20we1e-5lr10f.pt --source path/to/image.jpg

Outputs will be saved under runs/predict/<name>/ by default (Ultralytics behavior).
"""
from pathlib import Path
import argparse
from datetime import datetime

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLO predictions with a trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/jlee/Desktop/github/cv-training/models/beach_detection_100e30p20we1e-5lr10f.pt",
        help="Path to the trained model .pt file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/roi_filtered_data/test/images",
        help="Inference source: file, directory, URL, or stream",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size for inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id, 'cpu', or 'mps'")
    parser.add_argument("--save", action="store_true", help="Save visualized predictions")
    parser.add_argument(
        "--name",
        type=str,
        default=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Run name under runs/predict/",
    )
    parser.add_argument("--half", action="store_true", help="Use half precision (CUDA only)")

    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Run prediction
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        name=args.name,
        half=args.half,
        project="runs/predict",
    )

    # Print a short summary
    out_dir = Path("runs/predict") / args.name
    print(f"\nPrediction complete. Outputs: {out_dir.resolve()}")
    if len(results):
        r0 = results[0]
        print(f"First file: {getattr(r0, 'path', 'n/a')} -> {len(getattr(r0, 'boxes', []))} detections")


if __name__ == "__main__":
    main()
