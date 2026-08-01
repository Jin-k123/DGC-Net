"""Evaluation entry point for DGC-Net single-cell classification.

The script is aligned with ``train_github.py`` and supports repeated evaluation
across multiple random seeds, class-wise metrics, confusion matrices,
prediction export, and aggregate mean/std reporting.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.DGC_Net import DGC_Net
from Run.train.custom_dataloader import Custom_split_dataloader


DEFAULT_SEEDS = (24, 42, 3472)
DEFAULT_CLASS_NAMES = ("Bla", "Eos", "Lym", "Mon", "Neu", "RLym")
EPSILON = 1e-12


@dataclass(frozen=True)
class EvaluationPaths:
    output_dir: Path
    metrics_text: Path
    overall_metrics_csv: Path
    class_metrics_csv: Path
    predictions_csv: Path
    confusion_matrix_csv: Path
    confusion_matrix_percentage_csv: Path
    confusion_matrix_plot: Path


@dataclass
class EvaluationResult:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    macro_iou: float
    weighted_auc: float
    kappa: float
    class_precision: np.ndarray
    class_recall: np.ndarray
    class_f1: np.ndarray
    class_iou: np.ndarray
    class_auc: np.ndarray
    class_support: np.ndarray
    confusion_matrix: np.ndarray
    confusion_matrix_percentage: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray

    def overall_metrics(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_iou": self.macro_iou,
            "weighted_auc": self.weighted_auc,
            "kappa": self.kappa,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DGC-Net checkpoints on the test split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root directory used by the training and evaluation scripts.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="DGC-Net",
        help="Experiment name used to locate checkpoints and save results.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds whose checkpoints will be evaluated.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. This option requires exactly one seed.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model_best.pth",
        help="Checkpoint filename inside each seed directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Evaluation device, for example 'auto', 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument(
        "--batch-size",
        "--batchSize",
        dest="batch_size",
        type=int,
        default=32,
        help="Mini-batch size used during evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        "--threads",
        dest="num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--num-classes",
        "--num_classes",
        dest="num_classes",
        type=int,
        default=6,
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(DEFAULT_CLASS_NAMES),
        help="Class names in label-index order.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic cuDNN behavior when supported.",
    )
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the checkpoint state dictionary to match the model exactly.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_root}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2.")
    if args.image_size < 1:
        raise ValueError("--image-size must be positive.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size must be between 0 and 1.")
    if len(args.class_names) != args.num_classes:
        raise ValueError(
            "The number of --class-names entries must equal --num-classes."
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicate values.")
    if args.checkpoint is not None:
        if len(args.seeds) != 1:
            raise ValueError("--checkpoint requires exactly one value in --seeds.")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_evaluation_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def build_test_loader(
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> DataLoader:
    test_dataset = Custom_split_dataloader(
        str(args.data_root),
        transform=build_evaluation_transform(args.image_size),
        test_size=args.test_size,
        val_size=args.validation_size,
        train=False,
        test=True,
        val=False,
    )

    if len(test_dataset) == 0:
        raise RuntimeError("The test split is empty.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    print(f"Test samples: {len(test_dataset)}")
    return test_loader


def build_model(device: torch.device, num_classes: int) -> nn.Module:
    return DGC_Net(num_classes=num_classes).to(device)


def extract_logits(outputs: Any) -> Tensor:
    if isinstance(outputs, Tensor):
        return outputs

    if isinstance(outputs, dict):
        for key in ("logits", "logits_main"):
            value = outputs.get(key)
            if isinstance(value, Tensor):
                return value

        for value in outputs.values():
            if isinstance(value, Tensor) and value.ndim == 2:
                return value

        raise ValueError("No two-dimensional logits tensor was found in the output dictionary.")

    if isinstance(outputs, (tuple, list)):
        for value in outputs:
            if isinstance(value, Tensor) and value.ndim == 2:
                return value

        raise ValueError("No two-dimensional logits tensor was found in the output sequence.")

    raise TypeError(f"Unsupported model output type: {type(outputs).__name__}")


def normalize_state_dict_keys(state_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return dict(state_dict)


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    incompatible = model.load_state_dict(
        normalize_state_dict_keys(state_dict),
        strict=strict,
    )
    if not strict:
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys}")

    print(f"Loaded model weights from {checkpoint_path}")


def resolve_checkpoint_path(
    args: argparse.Namespace,
    seed: int,
) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint

    return (
        args.output_dir
        / "checkpoints"
        / args.experiment_name
        / f"seed_{seed}"
        / args.checkpoint_name
    )


def build_evaluation_paths(
    output_dir: Path,
    experiment_name: str,
    seed: int,
) -> EvaluationPaths:
    seed_output_dir = output_dir / "evaluation" / experiment_name / f"seed_{seed}"
    return EvaluationPaths(
        output_dir=seed_output_dir,
        metrics_text=seed_output_dir / "metrics.txt",
        overall_metrics_csv=seed_output_dir / "overall_metrics.csv",
        class_metrics_csv=seed_output_dir / "class_metrics.csv",
        predictions_csv=seed_output_dir / "predictions.csv",
        confusion_matrix_csv=seed_output_dir / "confusion_matrix.csv",
        confusion_matrix_percentage_csv=(
            seed_output_dir / "confusion_matrix_percentage.csv"
        ),
        confusion_matrix_plot=(
            seed_output_dir / "confusion_matrix_percentage.png"
        ),
    )


def compute_class_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    class_auc = np.full(num_classes, np.nan, dtype=np.float64)
    for class_index in range(num_classes):
        binary_labels = (labels == class_index).astype(np.int64)
        if np.unique(binary_labels).size < 2:
            continue
        class_auc[class_index] = roc_auc_score(
            binary_labels,
            probabilities[:, class_index],
        )
    return class_auc


def weighted_nan_average(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> EvaluationResult:
    model.eval()
    label_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []
    probability_batches: List[np.ndarray] = []

    progress = tqdm(test_loader, desc="Testing", ncols=100)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = extract_logits(model(images))
        if logits.ndim != 2 or logits.shape[1] != num_classes:
            raise ValueError(
                "Model logits must have shape [batch_size, num_classes], "
                f"but received {tuple(logits.shape)}."
            )

        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        label_batches.append(labels.cpu().numpy())
        prediction_batches.append(predictions.cpu().numpy())
        probability_batches.append(probabilities.cpu().numpy())

    if not label_batches:
        raise RuntimeError("The test DataLoader contains no batches.")

    all_labels = np.concatenate(label_batches).astype(np.int64, copy=False)
    all_predictions = np.concatenate(prediction_batches).astype(np.int64, copy=False)
    all_probabilities = np.concatenate(probability_batches).astype(np.float64, copy=False)

    label_indices = np.arange(num_classes)
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=label_indices,
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix_percentage = np.divide(
        matrix.astype(np.float64),
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals > 0,
    ) * 100.0

    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=label_indices,
            average=None,
            zero_division=0,
        )
    )

    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0).astype(np.float64) - true_positive
    false_negative = matrix.sum(axis=1).astype(np.float64) - true_positive
    class_iou = true_positive / (
        true_positive + false_positive + false_negative + EPSILON
    )
    class_auc = compute_class_auc(
        all_labels,
        all_probabilities,
        num_classes,
    )

    accuracy = float(np.mean(all_predictions == all_labels))
    macro_precision = float(np.mean(class_precision))
    macro_recall = float(np.mean(class_recall))
    macro_f1 = float(np.mean(class_f1))
    weighted_f1 = float(
        np.average(class_f1, weights=class_support)
        if class_support.sum() > 0
        else 0.0
    )
    macro_iou = float(np.mean(class_iou))
    weighted_auc = weighted_nan_average(class_auc, class_support.astype(np.float64))
    kappa = float(cohen_kappa_score(all_labels, all_predictions))

    return EvaluationResult(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_iou=macro_iou,
        weighted_auc=weighted_auc,
        kappa=kappa,
        class_precision=class_precision,
        class_recall=class_recall,
        class_f1=class_f1,
        class_iou=class_iou,
        class_auc=class_auc,
        class_support=class_support,
        confusion_matrix=matrix,
        confusion_matrix_percentage=matrix_percentage,
        labels=all_labels,
        predictions=all_predictions,
        probabilities=all_probabilities,
    )


def format_float(value: float, digits: int = 6) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"


def save_overall_metrics(
    result: EvaluationResult,
    seed: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result.overall_metrics()
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", *metrics.keys()])
        writer.writerow([seed, *[format_float(value) for value in metrics.values()]])


def save_class_metrics(
    result: EvaluationResult,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class_index",
                "class_name",
                "support",
                "precision",
                "recall",
                "f1",
                "iou",
                "auc",
            ]
        )
        for index, class_name in enumerate(class_names):
            writer.writerow(
                [
                    index,
                    class_name,
                    int(result.class_support[index]),
                    format_float(float(result.class_precision[index])),
                    format_float(float(result.class_recall[index])),
                    format_float(float(result.class_f1[index])),
                    format_float(float(result.class_iou[index])),
                    format_float(float(result.class_auc[index])),
                ]
            )


def save_predictions(
    result: EvaluationResult,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability_columns = [f"probability_{name}" for name in class_names]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "correct",
                *probability_columns,
            ]
        )
        for sample_index, (label, prediction, probabilities) in enumerate(
            zip(result.labels, result.predictions, result.probabilities)
        ):
            writer.writerow(
                [
                    sample_index,
                    int(label),
                    class_names[int(label)],
                    int(prediction),
                    class_names[int(prediction)],
                    int(label == prediction),
                    *[format_float(float(value)) for value in probabilities],
                ]
            )


def save_matrix_csv(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    percentage: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            if percentage:
                values = [f"{float(value):.4f}" for value in row]
            else:
                values = [int(value) for value in row]
            writer.writerow([class_name, *values])


def draw_confusion_matrix(
    matrix_percentage: np.ndarray,
    class_names: Sequence[str],
    accuracy: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_size = max(7.0, 0.9 * len(class_names) + 3.0)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size * 0.82))
    image = axis.imshow(matrix_percentage, interpolation="nearest", cmap="Blues")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Percentage (%)")

    threshold = matrix_percentage.max() / 2.0 if matrix_percentage.size else 0.0
    for row_index in range(matrix_percentage.shape[0]):
        for column_index in range(matrix_percentage.shape[1]):
            value = matrix_percentage[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Normalized Confusion Matrix (Accuracy = {accuracy * 100:.2f}%)",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_metrics_report(
    result: EvaluationResult,
    class_names: Sequence[str],
    seed: int,
    checkpoint_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Seed: {seed}",
        f"Checkpoint: {checkpoint_path}",
        f"Accuracy: {result.accuracy:.6f}",
        f"Macro Precision: {result.macro_precision:.6f}",
        f"Macro Recall: {result.macro_recall:.6f}",
        f"Macro F1: {result.macro_f1:.6f}",
        f"Weighted F1: {result.weighted_f1:.6f}",
        f"Macro IoU: {result.macro_iou:.6f}",
        f"Weighted AUC: {format_float(result.weighted_auc)}",
        f"Kappa: {result.kappa:.6f}",
        "",
        "Per-class metrics:",
    ]

    for index, class_name in enumerate(class_names):
        lines.append(
            f"{index} ({class_name}): "
            f"support={int(result.class_support[index])}, "
            f"precision={result.class_precision[index]:.6f}, "
            f"recall={result.class_recall[index]:.6f}, "
            f"f1={result.class_f1[index]:.6f}, "
            f"iou={result.class_iou[index]:.6f}, "
            f"auc={format_float(float(result.class_auc[index]))}"
        )

    lines.extend(["", "Confusion matrix (row-normalized percentage):"])
    for class_name, row in zip(class_names, result.confusion_matrix_percentage):
        row_text = ", ".join(f"{value:.2f}%" for value in row)
        lines.append(f"{class_name}: {row_text}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:11]))


def save_evaluation_outputs(
    result: EvaluationResult,
    paths: EvaluationPaths,
    class_names: Sequence[str],
    seed: int,
    checkpoint_path: Path,
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_report(
        result,
        class_names,
        seed,
        checkpoint_path,
        paths.metrics_text,
    )
    save_overall_metrics(result, seed, paths.overall_metrics_csv)
    save_class_metrics(result, class_names, paths.class_metrics_csv)
    save_predictions(result, class_names, paths.predictions_csv)
    save_matrix_csv(
        result.confusion_matrix,
        class_names,
        paths.confusion_matrix_csv,
        percentage=False,
    )
    save_matrix_csv(
        result.confusion_matrix_percentage,
        class_names,
        paths.confusion_matrix_percentage_csv,
        percentage=True,
    )
    draw_confusion_matrix(
        result.confusion_matrix_percentage,
        class_names,
        result.accuracy,
        paths.confusion_matrix_plot,
    )
    print(f"Evaluation outputs saved to {paths.output_dir}")


def run_one_seed(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    print(f"\n{'=' * 18} Seed {seed} {'=' * 18}\n")
    set_random_seed(seed, deterministic=args.deterministic)

    checkpoint_path = resolve_checkpoint_path(args, seed)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    test_loader = build_test_loader(args, device, seed)
    model = build_model(device, args.num_classes)
    load_model_weights(
        model,
        checkpoint_path,
        device,
        strict=args.strict_load,
    )

    result = evaluate(
        model,
        test_loader,
        device,
        args.num_classes,
    )
    paths = build_evaluation_paths(
        args.output_dir,
        args.experiment_name,
        seed,
    )
    save_evaluation_outputs(
        result,
        paths,
        args.class_names,
        seed,
        checkpoint_path,
    )
    return result.overall_metrics()


def save_summary(
    results: Mapping[int, Mapping[str, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = list(results.keys())
    metric_names = list(next(iter(results.values())).keys())

    rows: List[List[str]] = []
    for seed in seeds:
        rows.append(
            [
                str(seed),
                *[format_float(results[seed][metric]) for metric in metric_names],
            ]
        )

    metric_array = np.asarray(
        [[results[seed][metric] for metric in metric_names] for seed in seeds],
        dtype=np.float64,
    )
    mean_values = np.nanmean(metric_array, axis=0)
    std_values = np.nanstd(metric_array, axis=0, ddof=0)
    rows.append(["mean", *[format_float(value) for value in mean_values]])
    rows.append(["std", *[format_float(value) for value in std_values]])

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", *metric_names])
        writer.writerows(rows)

    print("\nAggregate results:")
    for metric, mean_value, std_value in zip(metric_names, mean_values, std_values):
        print(f"{metric}: {mean_value:.6f} +/- {std_value:.6f}")
    print(f"Summary saved to {output_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    results: Dict[int, Dict[str, float]] = {}
    for seed in args.seeds:
        results[seed] = run_one_seed(seed, args, device)

    summary_path = (
        args.output_dir
        / "evaluation"
        / args.experiment_name
        / "summary.csv"
    )
    save_summary(results, summary_path)


if __name__ == "__main__":
    main()
