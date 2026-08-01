"""Training entry point for DGC-Net single-cell classification.

The script supports repeated experiments with multiple random seeds, Mixup
training, class-wise evaluation, checkpointing, metric logging, and accuracy
curve visualization.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.DGC_Net import DGC_Net
from Run.custom_dataloader import Custom_split_dataloader
from Run.mixup import Mixup


DEFAULT_SEEDS = (24, 42, 3472)


@dataclass(frozen=True)
class ExperimentPaths:
    checkpoint_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    test_log: Path
    validation_log: Path
    accuracy_plot: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DGC-Net for single-cell image classification."
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
        help="Directory used for checkpoints, logs, and plots.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="DGC-Net",
        help="Name used to organize experiment outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds used for repeated experiments.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device, for example 'auto', 'cpu', 'cuda', or 'cuda:0'.",
    )
    parser.add_argument(
        "--batch-size",
        "--batchSize",
        dest="batch_size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument(
        "--num-workers",
        "--threads",
        dest="num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        help="Optional model weights used to initialize every seed run.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic cuDNN behavior when supported.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_root}")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
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
    if args.pretrained_weights is not None and not args.pretrained_weights.is_file():
        raise FileNotFoundError(
            f"Pretrained weights do not exist: {args.pretrained_weights}"
        )


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_paths(output_dir: Path, experiment_name: str, seed: int) -> ExperimentPaths:
    checkpoint_dir = output_dir / "checkpoints" / experiment_name / f"seed_{seed}"
    log_dir = output_dir / "logs" / experiment_name / f"seed_{seed}"
    plot_dir = output_dir / "plots" / experiment_name / f"seed_{seed}"

    return ExperimentPaths(
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=checkpoint_dir / "model_best.pth",
        last_checkpoint=checkpoint_dir / "model_last.pth",
        test_log=log_dir / "test.txt",
        validation_log=log_dir / "validation.txt",
        accuracy_plot=plot_dir / "accuracy.png",
    )


def build_transforms(
    image_size: int,
) -> Tuple[transforms.Compose, transforms.Compose]:
    base_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
    ]
    train_transform = transforms.Compose(
        [
            *base_transforms,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            *base_transforms,
            transforms.ToTensor(),
        ]
    )
    return train_transform, evaluation_transform


def build_dataset(
    data_root: Path,
    data_transform: transforms.Compose,
    test_size: float,
    validation_size: float,
    *,
    train: bool,
    test: bool,
    validation: bool,
):
    return Custom_split_dataloader(
        str(data_root),
        transform=data_transform,
        test_size=test_size,
        val_size=validation_size,
        train=train,
        test=test,
        val=validation,
    )


def build_dataloaders(
    args: argparse.Namespace,
    train_transform: transforms.Compose,
    evaluation_transform: transforms.Compose,
    device: torch.device,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("Loading datasets...")

    train_dataset = build_dataset(
        args.data_root,
        train_transform,
        args.test_size,
        args.validation_size,
        train=True,
        test=False,
        validation=False,
    )
    test_dataset = build_dataset(
        args.data_root,
        evaluation_transform,
        args.test_size,
        args.validation_size,
        train=False,
        test=True,
        validation=False,
    )
    validation_dataset = build_dataset(
        args.data_root,
        evaluation_transform,
        args.test_size,
        args.validation_size,
        train=False,
        test=False,
        validation=True,
    )

    if min(len(train_dataset), len(test_dataset), len(validation_dataset)) == 0:
        raise RuntimeError("At least one dataset split is empty.")

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    generator = torch.Generator()
    generator.manual_seed(seed)

    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    print(
        "Dataset sizes: "
        f"train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}, "
        f"test={len(test_dataset)}"
    )
    return train_loader, test_loader, validation_loader


def build_model(device: torch.device, num_classes: int) -> nn.Module:
    return DGC_Net(num_classes=num_classes).to(device)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    return optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )


def build_mixup(
    num_classes: int,
    mixup_alpha: float,
    label_smoothing: float,
) -> Optional[Mixup]:
    if mixup_alpha <= 0.0:
        return None

    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=0.0,
        switch_prob=0.0,
        mode="batch",
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )


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


def load_model_weights(model: nn.Module, weights_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {weights_path}")

    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded model weights from {weights_path}")


def save_checkpoint(model: nn.Module, path: Path) -> None:
    ensure_parent(path)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")


def write_metrics_to_file(
    epoch: int,
    overall_accuracy: float,
    class_accuracies: Sequence[float],
    file_path: Path,
) -> None:
    ensure_parent(file_path)
    class_metrics = ", ".join(
        f"Class{index}_Acc: {accuracy:.4f}"
        for index, accuracy in enumerate(class_accuracies)
    )
    with file_path.open("a", encoding="utf-8") as file:
        file.write(
            f"Epoch: {epoch}, Overall Acc: {overall_accuracy:.4f}, "
            f"{class_metrics}\n"
        )


def plot_accuracy_curves(
    test_accuracies: Sequence[float],
    validation_accuracies: Sequence[float],
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    epochs = range(1, len(test_accuracies) + 1)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, test_accuracies, label="Test")
    axis.plot(epochs, validation_accuracies, label="Validation")
    axis.set_title("Accuracy Curves")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    mixup_fn: Optional[Mixup] = None,
) -> float:
    model.train()
    running_loss = 0.0

    progress = tqdm(train_loader, total=len(train_loader), ncols=100)
    for step, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        loss_targets: Tensor
        if mixup_fn is not None:
            inputs, loss_targets = mixup_fn(inputs, targets)
        else:
            loss_targets = targets

        optimizer.zero_grad(set_to_none=True)
        logits = extract_logits(model(inputs))
        loss = F.cross_entropy(logits, loss_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        average_loss = running_loss / step
        progress.set_description(f"Epoch [{epoch}/{total_epochs}]")
        progress.set_postfix(loss=f"{average_loss:.4f}")

    if len(train_loader) == 0:
        raise RuntimeError("The training DataLoader contains no batches.")

    return running_loss / len(train_loader)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epoch: int,
    split_name: str,
    log_path: Path,
) -> Tuple[float, List[float]]:
    model.eval()
    total_samples = 0
    total_correct = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = extract_logits(model(images))
        predictions = logits.argmax(dim=1)

        total_samples += labels.numel()
        total_correct += predictions.eq(labels).sum().item()

        labels_cpu = labels.detach().cpu()
        predictions_cpu = predictions.detach().cpu()
        class_total += torch.bincount(labels_cpu, minlength=num_classes)[:num_classes]
        correct_labels = labels_cpu[predictions_cpu.eq(labels_cpu)]
        class_correct += torch.bincount(
            correct_labels, minlength=num_classes
        )[:num_classes]

    if total_samples == 0:
        raise RuntimeError(f"The {split_name} DataLoader contains no samples.")

    class_accuracies = [
        class_correct[index].item() / class_total[index].item()
        if class_total[index].item() > 0
        else 0.0
        for index in range(num_classes)
    ]
    overall_accuracy = total_correct / total_samples
    class_summary = ", ".join(
        f"{index}_Acc:{accuracy:.4f}"
        for index, accuracy in enumerate(class_accuracies)
    )

    print(
        f"Epoch [{epoch}] {split_name} Acc: {overall_accuracy:.4f}, "
        f"{class_summary}"
    )
    write_metrics_to_file(
        epoch,
        overall_accuracy,
        class_accuracies,
        log_path,
    )
    return overall_accuracy, class_accuracies


def run_one_seed(
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    print(f"\n{'=' * 18} Seed {seed} {'=' * 18}\n")
    set_random_seed(seed, deterministic=args.deterministic)

    train_transform, evaluation_transform = build_transforms(args.image_size)
    train_loader, test_loader, validation_loader = build_dataloaders(
        args,
        train_transform,
        evaluation_transform,
        device,
        seed,
    )

    model = build_model(device, args.num_classes)
    optimizer = build_optimizer(args, model)
    mixup_fn = build_mixup(
        args.num_classes,
        args.mixup_alpha,
        args.label_smoothing,
    )
    paths = build_paths(args.output_dir, args.experiment_name, seed)

    if args.pretrained_weights is not None:
        load_model_weights(model, args.pretrained_weights, device)

    test_history: List[float] = []
    validation_history: List[float] = []
    best_validation_accuracy = float("-inf")
    test_accuracy_at_best_validation = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            mixup_fn=mixup_fn,
        )
        print(f"Epoch [{epoch}] Train Loss: {train_loss:.4f}")

        test_accuracy, _ = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            num_classes=args.num_classes,
            epoch=epoch,
            split_name="Test",
            log_path=paths.test_log,
        )
        validation_accuracy, _ = evaluate(
            model=model,
            data_loader=validation_loader,
            device=device,
            num_classes=args.num_classes,
            epoch=epoch,
            split_name="Validation",
            log_path=paths.validation_log,
        )

        test_history.append(test_accuracy)
        validation_history.append(validation_accuracy)
        plot_accuracy_curves(
            test_history,
            validation_history,
            paths.accuracy_plot,
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            test_accuracy_at_best_validation = test_accuracy
            best_epoch = epoch
            save_checkpoint(model, paths.best_checkpoint)
            print(
                f"New best checkpoint at epoch {epoch}: "
                f"validation={best_validation_accuracy:.4f}, "
                f"test={test_accuracy_at_best_validation:.4f}"
            )

        save_checkpoint(model, paths.last_checkpoint)

    summary = {
        "seed": float(seed),
        "best_epoch": float(best_epoch),
        "best_validation_accuracy": best_validation_accuracy,
        "test_accuracy_at_best_validation": test_accuracy_at_best_validation,
    }

    print(f"\nTraining completed for seed {seed}.")
    print(
        f"Best validation accuracy: {best_validation_accuracy:.4f} "
        f"at epoch {best_epoch}"
    )
    print(
        "Test accuracy at the best validation epoch: "
        f"{test_accuracy_at_best_validation:.4f}"
    )
    return summary


def print_final_summary(results: Iterable[Dict[str, float]]) -> None:
    results = list(results)
    if not results:
        return

    print(f"\n{'=' * 18} Final Summary {'=' * 18}")
    for result in results:
        print(
            f"Seed {int(result['seed'])}: "
            f"best_epoch={int(result['best_epoch'])}, "
            f"best_val={result['best_validation_accuracy']:.4f}, "
            f"test_at_best_val={result['test_accuracy_at_best_validation']:.4f}"
        )

    test_scores = np.asarray(
        [result["test_accuracy_at_best_validation"] for result in results],
        dtype=np.float64,
    )
    print(
        "Test accuracy across seeds: "
        f"{test_scores.mean():.4f} ± {test_scores.std(ddof=0):.4f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)

    print(f"Arguments: {args}")
    print(f"Using device: {device}")

    results = [
        run_one_seed(seed=seed, args=args, device=device)
        for seed in args.seeds
    ]
    print_final_summary(results)


if __name__ == "__main__":
    main()
