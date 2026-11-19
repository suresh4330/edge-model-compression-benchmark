"""
Evaluate the four EDGE-BENCH checkpoints on the CIFAR-100 test split.

Run from the repository root:
    python evaluate_f1.py

Outputs:
    results/f1_evaluation.json
    results/f1_evaluation.csv

Reports accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1.
The script does not train, modify, or overwrite model checkpoints.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.main import get_model


MODEL_IDS = ["baseline", "int8", "pruned", "kd"]

NORMALIZE = transforms.Normalize(
    mean=(0.5071, 0.4865, 0.4409),
    std=(0.2673, 0.2564, 0.2762),
)


def evaluate_model(model_id, loader, class_names):
    model = get_model(model_id)
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in loader:
            if model_id == "int8":
                images = torch.quantize_per_tensor(
                    images,
                    float(model.input_scale),
                    int(model.input_zero_point),
                    torch.quint8,
                )

            logits = model(images)

            if logits.is_quantized:
                logits = logits.dequantize()

            predictions = logits.argmax(dim=1)
            all_targets.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    report = classification_report(
        all_targets,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_id": model_id,
        "status": "success",
        "test_images": len(all_targets),
        "accuracy_percent": round(accuracy_score(all_targets, all_predictions) * 100, 4),
        "macro_f1_percent": round(f1_score(
            all_targets, all_predictions, average="macro", zero_division=0
        ) * 100, 4),
        "weighted_f1_percent": round(f1_score(
            all_targets, all_predictions, average="weighted", zero_division=0
        ) * 100, 4),
        "per_class_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 Macro-F1 evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_IDS,
        default=MODEL_IDS,
        help="Models to evaluate (default: all four)",
    )
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE,
    ])

    test_dataset = datasets.CIFAR100(
        root=os.path.join(ROOT_DIR, "data"),
        train=False,
        download=False,
        transform=test_transform,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    class_names = test_dataset.classes
    results = []

    print(f"CIFAR-100 test images: {len(test_dataset)}")
    print(f"Evaluating: {', '.join(args.models)}")

    for model_id in args.models:
        print(f"\n[{model_id}] evaluating...")
        try:
            result = evaluate_model(model_id, loader, class_names)
            print(
                f"Accuracy: {result['accuracy_percent']:.2f}% | "
                f"Macro-F1: {result['macro_f1_percent']:.2f}% | "
                f"Weighted-F1: {result['weighted_f1_percent']:.2f}%"
            )
        except Exception as exc:
            result = {
                "model_id": model_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"FAILED: {result['error']}")
        results.append(result)

    os.makedirs("results", exist_ok=True)
    payload = {
        "dataset": "CIFAR-100",
        "split": "official test split",
        "test_images": len(test_dataset),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_definition": {
            "macro_f1": "Unweighted mean of per-class F1 across all 100 classes.",
            "weighted_f1": "Per-class F1 averaged using each class's support.",
            "accuracy": "Correct predictions divided by all test images.",
        },
        "results": results,
    }

    json_path = os.path.join(ROOT_DIR, "results", "f1_evaluation.json")
    csv_path = os.path.join(ROOT_DIR, "results", "f1_evaluation.csv")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model_id", "status", "test_images",
                "accuracy_percent", "macro_f1_percent",
                "weighted_f1_percent", "error",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print("Check each model status before using its F1 score in a paper.")


if __name__ == "__main__":
    main()
