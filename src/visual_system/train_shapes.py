"""Train the optional multi-task neural shape classifier from labeled SQLite samples."""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from visual_system.services.neural_shape_model import (
    CHECKPOINT_VERSION,
    IMAGE_SIZE,
    MIN_DEPLOYMENT_ACCURACY,
    MIN_TRAIN_SAMPLES_PER_CLASS,
    MIN_VALIDATION_SAMPLES_PER_CLASS,
    OUTLINE_CLASSES,
    SOLID_CLASSES,
    SOLID_LABEL_ALIASES,
    SOLID_TO_OUTLINE,
    _torch_modules,
    prepare_image,
)


def load_samples(database: Path, validation_fraction: float, seed: int):
    root = database.parent
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT label, image_path FROM shape_samples WHERE image_path IS NOT NULL ORDER BY id"
        ).fetchall()
    grouped = defaultdict(list)
    for label, stored_path in rows:
        label = SOLID_LABEL_ALIASES.get(label, label)
        if label not in (*OUTLINE_CLASSES, *SOLID_TO_OUTLINE):
            continue
        image_path = (root / stored_path).resolve()
        if not image_path.is_relative_to(root.resolve()) or not image_path.is_file():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            grouped[label].append(prepare_image(image))
    if len(grouped) < 2:
        raise ValueError("Cần ảnh đã gán nhãn ở ít nhất hai lớp để huấn luyện")
    rng = random.Random(seed)
    train, validation = [], []
    for label, images in sorted(grouped.items()):
        rng.shuffle(images)
        validation_count = (
            min(len(images) - 1, max(1, round(len(images) * validation_fraction)))
            if len(images) > 1
            else 0
        )
        split = len(images) - validation_count
        train.extend((label, image) for image in images[:split])
        validation.extend((label, image) for image in images[split:])
    if not validation:
        raise ValueError("Cần tối thiểu hai ảnh ở một hoặc nhiều lớp để tạo validation holdout")
    return train, validation, {label: len(items) for label, items in sorted(grouped.items())}


def _augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(-18, 18))
    scale = float(rng.uniform(0.88, 1.12))
    shift_x, shift_y = (float(item) for item in rng.uniform(-5, 5, size=2))
    matrix = cv2.getRotationMatrix2D((IMAGE_SIZE / 2, IMAGE_SIZE / 2), angle, scale)
    matrix[:, 2] += (shift_x, shift_y)
    augmented = cv2.warpAffine(image, matrix, (IMAGE_SIZE, IMAGE_SIZE), borderValue=(18, 18, 18))
    if rng.random() < 0.5:
        augmented = cv2.flip(augmented, 1)
    if rng.random() < 0.15:
        augmented = cv2.flip(augmented, 0)
    brightness = float(rng.uniform(0.82, 1.18))
    contrast = float(rng.uniform(0.85, 1.15))
    adjusted = (augmented.astype(np.float32) - 127.5) * contrast + 127.5
    return np.clip(adjusted * brightness, 0, 255).astype(np.uint8)


def _to_tensor(torch, image: np.ndarray):
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255)


def _classification_metrics(confusion: Counter, classes: tuple[str, ...]) -> dict:
    metrics = {}
    total = 0
    correct = 0
    for label in classes:
        true_positive = confusion[(label, label)]
        support = sum(confusion[(label, predicted)] for predicted in classes)
        predicted_count = sum(confusion[(actual, label)] for actual in classes)
        total += support
        correct += true_positive
        metrics[label] = {
            "count": support,
            "precision": round(true_positive / max(predicted_count, 1), 4),
            "recall": round(true_positive / max(support, 1), 4),
        }
    return {
        "accuracy": round(correct / max(total, 1), 4),
        "count": total,
        "classes": metrics,
    }


def _print_solid_metrics_table(
    sample_counts: dict,
    metrics: dict,
    deployment_report: dict,
) -> None:
    solid_metrics = metrics["solid_classes"]
    solid_report = deployment_report["heads"]["solid"]
    outline_count = sum(sample_counts.get(label, 0) for label in OUTLINE_CLASSES)

    print("\nBest checkpoint: solid-head validation by class")
    print(
        f"{'Class':<10} | {'Train':>5} | {'Val':>5} | {'Precision':>9} | "
        f"{'Recall':>7} | {'Gate':>4}"
    )
    print("-" * 61)
    for label in SOLID_CLASSES:
        total = outline_count if label == "none" else sample_counts.get(label, 0)
        validation_count = solid_metrics[label]["count"]
        train_count = max(total - validation_count, 0)
        class_failures = [
            item for item in solid_report["failures"] if item.startswith(f"{label}:")
        ]
        gate = "PASS" if not class_failures else "FAIL"
        item = solid_metrics[label]
        print(
            f"{label:<10} | {train_count:>5} | {validation_count:>5} | "
            f"{item['precision']:>8.1%} | {item['recall']:>6.1%} | {gate:>4}"
        )
    overall_gate = "PASS" if solid_report["deployable"] else "FAIL"
    print("-" * 61)
    print(
        f"Solid accuracy: {metrics['solid_accuracy']:.1%} "
        f"(required {MIN_DEPLOYMENT_ACCURACY:.0%}) | Deployment gate: {overall_gate}"
    )


def _deployment_report(train_counts: dict, validation_metrics: dict) -> dict:
    outline_counts = {
        label: train_counts.get(label, 0) for label in OUTLINE_CLASSES
    }
    solid_counts = {
        "none": sum(outline_counts.values()),
        **{label: train_counts.get(label, 0) for label in SOLID_CLASSES if label != "none"},
    }
    results = {}
    for head, classes, supports, head_metrics in (
        (
            "outline",
            OUTLINE_CLASSES,
            outline_counts,
            validation_metrics.get("outline", {}),
        ),
        (
            "solid",
            SOLID_CLASSES,
            solid_counts,
            validation_metrics.get("solid", {}),
        ),
    ):
        class_metrics = head_metrics.get("classes", {})
        failures = []
        for label in classes:
            item = class_metrics.get(label, {})
            if supports[label] < MIN_TRAIN_SAMPLES_PER_CLASS:
                failures.append(
                    f"{label}: train mẫu {supports[label]} "
                    f"< {MIN_TRAIN_SAMPLES_PER_CLASS}"
                )
            if item.get("count", 0) < MIN_VALIDATION_SAMPLES_PER_CLASS:
                failures.append(
                    f"{label}: validation mẫu {item.get('count', 0)} "
                    f"< {MIN_VALIDATION_SAMPLES_PER_CLASS}"
                )
            if item.get("precision", 0) < MIN_DEPLOYMENT_ACCURACY:
                failures.append(f"{label}: precision < {MIN_DEPLOYMENT_ACCURACY:.0%}")
            if item.get("recall", 0) < MIN_DEPLOYMENT_ACCURACY:
                failures.append(f"{label}: recall < {MIN_DEPLOYMENT_ACCURACY:.0%}")
        if head_metrics.get("accuracy", 0) < MIN_DEPLOYMENT_ACCURACY:
            failures.append(f"accuracy tổng < {MIN_DEPLOYMENT_ACCURACY:.0%}")
        results[head] = {"deployable": not failures, "failures": failures}
    return {
        "deployable_heads": [head for head, result in results.items() if result["deployable"]],
        "heads": results,
    }


def train(
    database: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    validation_fraction: float,
):
    torch, model_type = _torch_modules()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_samples, validation_samples, counts = load_samples(database, validation_fraction, seed)
    train_counts = Counter(label for label, _ in train_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_type().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    outline_counts = Counter(label for label, _ in train_samples if label in OUTLINE_CLASSES)
    solid_counts = Counter(label for label, _ in train_samples if label in SOLID_TO_OUTLINE)
    solid_supervised = [
        label
        for label, _ in train_samples
        if label in SOLID_TO_OUTLINE
    ]
    flat_sample_count = len(train_samples) - len(solid_supervised)

    def class_weights(classes, counts_by_class):
        values = [1 / max(counts_by_class.get(name, 0), 1) ** 0.5 for name in classes]
        values = np.asarray(values, dtype=np.float32)
        values /= max(float(values.mean()), 1e-6)
        return torch.tensor(values, dtype=torch.float32, device=device)

    outline_weights = class_weights(OUTLINE_CLASSES, outline_counts)
    solid_weights = class_weights(
        SOLID_CLASSES,
        {"none": flat_sample_count, **solid_counts},
    )
    outline_criterion = torch.nn.CrossEntropyLoss(weight=outline_weights, label_smoothing=0.03)
    solid_criterion = torch.nn.CrossEntropyLoss(weight=solid_weights, label_smoothing=0.03)
    outline_index = {label: index for index, label in enumerate(OUTLINE_CLASSES)}
    solid_index = {label: index for index, label in enumerate(SOLID_CLASSES)}
    augmentation_rng = np.random.default_rng(seed)
    best_accuracy = -1.0
    best_state = None
    best_metrics = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_samples))
        total_loss = 0.0
        total_items = 0
        for start in range(0, len(order), batch_size):
            batch = [train_samples[index] for index in order[start : start + batch_size]]
            images = torch.stack(
                [_to_tensor(torch, _augment(image, augmentation_rng)) for _, image in batch]
            ).to(device)
            outline_mask = torch.tensor(
                [label in OUTLINE_CLASSES for label, _ in batch], device=device
            )
            solid_mask = torch.ones(len(batch), dtype=torch.bool, device=device)
            outline_targets = torch.tensor(
                [outline_index[SOLID_TO_OUTLINE.get(label, label)] for label, _ in batch],
                device=device,
            )
            solid_targets = torch.tensor(
                [solid_index[label if label in SOLID_TO_OUTLINE else "none"] for label, _ in batch],
                device=device,
            )
            outline_logits, solid_logits = model(images)
            loss = outline_logits.sum() * 0 + solid_logits.sum() * 0
            if outline_mask.any():
                loss = loss + outline_criterion(
                    outline_logits[outline_mask], outline_targets[outline_mask]
                )
            if solid_mask.any():
                loss = loss + solid_criterion(
                    solid_logits[solid_mask], solid_targets[solid_mask]
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_items += len(batch)

        model.eval()
        sample_count = 0
        validation_loss = 0.0
        outline_confusion = Counter()
        solid_confusion = Counter()
        with torch.inference_mode():
            for start in range(0, len(validation_samples), batch_size):
                batch = validation_samples[start : start + batch_size]
                images = torch.stack([_to_tensor(torch, image) for _, image in batch]).to(device)
                outline_targets = torch.tensor(
                    [outline_index[SOLID_TO_OUTLINE.get(label, label)] for label, _ in batch],
                    device=device,
                )
                solid_targets = torch.tensor(
                    [
                        solid_index[label if label in SOLID_TO_OUTLINE else "none"]
                        for label, _ in batch
                    ],
                    device=device,
                )
                outline_logits, solid_logits = model(images)
                outline_mask = torch.tensor(
                    [label in OUTLINE_CLASSES for label, _ in batch], device=device
                )
                solid_mask = torch.ones(len(batch), dtype=torch.bool, device=device)
                batch_loss = outline_logits.sum() * 0 + solid_logits.sum() * 0
                if outline_mask.any():
                    batch_loss = batch_loss + outline_criterion(
                        outline_logits[outline_mask], outline_targets[outline_mask]
                    )
                    for target, predicted in zip(
                        outline_targets[outline_mask].tolist(),
                        outline_logits[outline_mask].argmax(1).tolist(),
                    ):
                        outline_confusion[
                            (OUTLINE_CLASSES[target], OUTLINE_CLASSES[predicted])
                        ] += 1
                if solid_mask.any():
                    batch_loss = batch_loss + solid_criterion(
                        solid_logits[solid_mask], solid_targets[solid_mask]
                    )
                    for target, predicted in zip(
                        solid_targets.tolist(), solid_logits.argmax(1).tolist()
                    ):
                        solid_confusion[
                            (SOLID_CLASSES[target], SOLID_CLASSES[predicted])
                        ] += 1
                validation_loss += float(batch_loss.item()) * len(batch)
                sample_count += len(batch)
        outline_metrics = _classification_metrics(outline_confusion, OUTLINE_CLASSES)
        solid_metrics = _classification_metrics(solid_confusion, SOLID_CLASSES)
        outline_macro = float(
            np.mean(
                [item["recall"] for item in outline_metrics["classes"].values() if item["count"]]
            )
        ) if outline_metrics["count"] else 0.0
        solid_macro = float(
            np.mean(
                [item["recall"] for item in solid_metrics["classes"].values() if item["count"]]
            )
        ) if solid_metrics["count"] else 0.0
        accuracy = (outline_macro + solid_macro) / 2
        record = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(total_items, 1), 5),
            "validation_loss": round(validation_loss / max(sample_count, 1), 5),
            "outline_accuracy": outline_metrics["accuracy"],
            "solid_accuracy": solid_metrics["accuracy"],
            "outline_macro_recall": round(outline_macro, 4),
            "solid_macro_recall": round(solid_macro, 4),
            "outline_classes": outline_metrics["classes"],
            "solid_classes": solid_metrics["classes"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = record

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": best_state,
        "outline_classes": list(OUTLINE_CLASSES),
        "solid_classes": list(SOLID_CLASSES),
        "image_size": IMAGE_SIZE,
        "sample_counts": counts,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "best_validation_balanced_macro_recall": best_accuracy,
        "validation_metrics": best_metrics,
        "history": history,
    }
    report = _deployment_report(
        train_counts,
        {
            "outline": {
                "accuracy": best_metrics["outline_accuracy"],
                "classes": best_metrics["outline_classes"],
            },
            "solid": {
                "accuracy": best_metrics["solid_accuracy"],
                "classes": best_metrics["solid_classes"],
            },
        },
    )
    checkpoint["deployment_report"] = report
    temporary_output = output.with_name(f"{output.name}.tmp")
    try:
        torch.save(checkpoint, temporary_output)
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "device": str(device),
                "samples": counts,
                "validation_metrics": best_metrics,
                "deployment_report": report,
            },
            ensure_ascii=False,
        )
    )
    _print_solid_metrics_table(counts, best_metrics, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/shape_library.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("data/models/shape_net.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.validation_fraction < 0.5:
        parser.error("epochs/batch-size phải dương và validation-fraction nằm trong (0, 0.5)")
    try:
        train(
            args.database,
            args.output,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
            args.validation_fraction,
        )
    except (ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
