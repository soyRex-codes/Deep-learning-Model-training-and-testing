"""Train and evaluate the exact three-layer CNN required by CSC725 HW2.

The file has five parts: model definition, hand-written complexity calculations,
data loading, training/evaluation, and the command-line entry point. A run
saves the measured results, graphs, and final checkpoint in its output directory.
The existing ``results/`` directory contains the recorded experiment.

To rerun without overwriting it:
    python hw2_solution.py --epochs 10 --output-dir results_new
The bonus FLOP count follows the homework's rule: one multiplication = one FLOP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


"""
Name: Rajkumar Kushwaha
Assignment: HW02
course: CSC725 (Computer Vision)
"""

class ThreeLayerCNN(nn.Module):
    """Three Conv-ReLU-Pool blocks followed by one ten-class linear layer."""

    def __init__(self) -> None:
        super().__init__()
        # CIFAR-10 images are RGB, so the first convolution receives 3 channels.
        # Padding 1 keeps height and width unchanged for every 3x3 convolution.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Three pools halve each spatial side: 32 -> 16 -> 8 -> 4.
        # The last activation volume is therefore 64 x 4 x 4 = 1,024 values.
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch [N, 3, 32, 32] to ten class scores per image [N, 10]."""
        # Each block extracts features, adds nonlinearity, then halves H and W.
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        # Return logits, not probabilities: CrossEntropyLoss handles log-softmax.
        return self.fc(self.flatten(x))


def count_parameters(model: nn.Module, *, verbose: bool = True) -> dict[str, Any]:
    """Count trainable weights and biases, by layer and in total.

    For a biased convolution: Cout * (Cin * Kh * Kw + 1). For a biased linear
    layer: Dout * (Din + 1). Tensor.numel() applies the same rule to the actual
    model tensors. Count shared tensors once and exclude frozen tensors because
    the question asks for *learnable* parameters.
    """
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name, layer in model.named_modules():
        counts = {"weights": 0, "biases": 0, "other": 0}
        # recurse=False avoids counting a child layer's parameters again in
        # its parent; seen also handles a tensor shared by multiple layers.
        for pname, parameter in layer.named_parameters(recurse=False):
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                category = "weights" if pname == "weight" else "biases" if pname == "bias" else "other"
                counts[category] += parameter.numel()
        total = sum(counts.values())
        if total:
            rows.append({"layer": name or "model", "type": type(layer).__name__, **counts, "total": total})
    result = {"layers": rows, "total": sum(row["total"] for row in rows)}
    if verbose:
        print("\nLearnable parameters (weights + biases):")
        for row in rows:
            print(f"  {row['layer']:8s}: {row['weights']:,} weights + {row['biases']:,} biases"
                  f" + {row['other']:,} other = {row['total']:,}")
        print(f"  Total: {result['total']:,}")
    return result


def calculate_flops(
    model: nn.Module, input_size: tuple[int, ...] = (1, 3, 32, 32), *, verbose: bool = True
) -> dict[str, Any]:
    """Estimate convolution/linear multiplications from actual output shapes.

    Accept C,H,W or N,C,H,W; report both per-image and whole-batch totals.
    A forward hook observes each layer's *actual* output dimensions. Conv2d
    costs Hout*Wout*Cout*(Cin/groups)*Kh*Kw multiplications; Linear costs
    Din*Dout. Ignore additions, biases, ReLU, and pooling. The
    dummy forward pass never updates weights or leaves hooks attached.
    """
    shape = tuple(input_size)
    # If the caller omits N, calculate the cost for one image.
    if len(shape) == 3:
        shape = (1, *shape)
    if len(shape) != 4 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("input_size must contain positive integers in C,H,W or N,C,H,W order")
    batch_size = shape[0]
    parameters = list(model.parameters())
    # The dummy input must live on the same device and have the same dtype as
    # the model (for example, both CPU float32 or both GPU float32).
    device = parameters[0].device if parameters else torch.device("cpu")
    dtype = parameters[0].dtype if parameters else torch.float32
    # FLOP counting should not permanently change train/eval settings.
    modes = {module: module.training for module in model.modules()}
    rows: list[dict[str, Any]] = []
    handles = []

    def make_hook(name: str):
        def hook(layer: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(layer, nn.Conv2d):
                kh, kw = layer.kernel_size
                # One output value multiplies each input-channel kernel value.
                # Divide by groups for grouped convolutions; ordinary CNN
                # convolutions here have groups=1.
                per_output = (layer.in_channels // layer.groups) * kh * kw
                batch_flops = output.numel() * per_output
            elif isinstance(layer, nn.Linear):
                batch_flops = output.numel() * layer.in_features
            else:
                return
            rows.append({
                "layer": name or "model", "type": type(layer).__name__,
                "input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                # output.numel() included N; divide to report one-image cost.
                "flops": batch_flops // batch_size, "batch_flops": batch_flops,
            })
        return hook

    try:
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                handles.append(layer.register_forward_hook(make_hook(name)))
        # No gradients or parameter updates are needed just to observe shapes.
        model.eval()
        with torch.no_grad():
            model(torch.zeros(shape, device=device, dtype=dtype))
    finally:
        # This runs even if an invalid input makes the forward pass fail.
        for handle in handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training
    result = {
        "input_size": list(shape), "layers": rows,
        "total_flops": sum(row["flops"] for row in rows),
        "batch_flops": sum(row["batch_flops"] for row in rows),
        "convention": "one multiplication = one FLOP; additions, biases, ReLU and pooling ignored",
    }
    if verbose:
        print("\nMultiplications per image (homework FLOP convention):")
        for row in rows:
            print(f"  {row['layer']:8s}: output {row['output_shape']} -> {row['flops']:,}")
        print(f"  Total per image: {result['total_flops']:,}")
    return result


@dataclass
class TrainConfig: # configuration class for training hyperparameters and paths
    # in simple terms hyperparameters are the settings that control how the model is trained and the path to the data and output directories
    """Hyperparameters and paths shared by the run and its saved checkpoint."""

    epochs: int = 10
    batch_size: int = 128 # 128 data sample each batch
    learning_rate: float = 0.001
    seed: int = 725 # this is the random seed for reproducibility helps ensure consistent results forking and restarting experiments
    # it is important to set the seed before creating the model and data loaders, seed is used to initialize all random number generators
    num_workers: int = 0
    cpu_threads: int = 4
    device: str = "auto"
    data_dir: str = "data"
    output_dir: str = "results"

    def validate(self) -> None:
        """ensuring training proceeds safely."""
        if self.epochs < 10:
            raise ValueError("HW2 requires at least 10 epochs")
        if self.batch_size <= 0 or self.num_workers < 0 or self.cpu_threads < 1:
            raise ValueError("Invalid batch size, worker count, or CPU thread count")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")


def choose_device(requested: str = "auto") -> torch.device:
    """Use CUDA or Apple MPS when available; otherwise train on CPU.""" # chooses the best available device for training
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before model and loader creation."""
    # why? seeding ensures that the random number generators are initialized with a known value, making the results reproducible.
    # eg: if we set the seed to 725, every time we run the code, the random numbers generated will be the same, leading to reproducible results.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def make_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Load the official CIFAR-10 splits and return training/test mini-batches."""
    # ToTensor converts an image from 0..255 bytes to a [3, 32, 32] tensor
    # in [0, 1]. Normalize applies (value - 0.5) / 0.5 in each RGB channel.
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    # Keep the official splits separate: only train=True supplies weight updates.
    # download=True reuses the local copy when CIFAR-10 is already present.
    training = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=transform)
    testing = datasets.CIFAR10(root=config.data_dir, train=False, download=True, transform=transform)
    # A seeded generator makes the shuffled training order reproducible.
    generator = torch.Generator().manual_seed(config.seed)
    # DataLoader yields batches of (images, labels). Keep the shorter final
    # batch so every example contributes to training and testing.
    common = {"batch_size": config.batch_size, "num_workers": config.num_workers, "drop_last": False}
    # Shuffle each training pass, but preserve test order for saved predictions.
    train_loader = DataLoader(training, shuffle=True, generator=generator, **common)
    test_loader = DataLoader(testing, shuffle=False, **common)
    return train_loader, test_loader


def train_one_epoch( # trains the model for one epoch over the given data loader, epoch means one complete pass through the entire dataset
        # updating the model's parameters after each batch
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> dict[str, float | int]:
    """Train over every batch once and return sample-weighted online metrics."""
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        # images: [batch, 3, 32, 32]; labels: [batch] class indices 0..9.
        images, labels = images.to(device), labels.to(device)
        # Gradients accumulate by default in PyTorch, so clear the previous
        # batch's gradients before computing this batch's new gradients.
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite")
        # Backpropagation computes parameter gradients; Adam then updates them.
        loss.backward()
        optimizer.step()
        size = labels.size(0)
        # CrossEntropyLoss returns a batch mean. Weight it by batch size so the
        # shorter last batch is not given the same weight as a full batch.
        total_loss += loss.item() * size
        # argmax chooses the highest-scoring class. These logits were produced
        # before the Adam update, so accuracy is an online training metric.
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += size
        if batch_index % 100 == 0:
            print(f"    batch {batch_index}/{len(loader)}; mean loss {total_loss/count:.4f}", flush=True)
    if count == 0:
        raise ValueError("The training loader is empty")
    # These are online epoch metrics measured before each batch's weight update.
    return {"loss": total_loss / count, "accuracy": correct / count, "correct": correct, "samples": count}


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
) -> dict[str, Any]:
    """Measuring loss/accuracy without learning and also retaining all test predictions for final report."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    predictions: list[int] = []
    targets: list[int] = []
    # Evaluation must not build a gradient graph or update model parameters.
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Evaluation loss became non-finite")
            size = labels.size(0)
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * size
            correct += (pred == labels).sum().item()
            count += size
            # Moving IDs back to CPU for a plain CSV of every test prediction.
            predictions.extend(pred.cpu().tolist())
            targets.extend(labels.cpu().tolist())
    if count == 0:
        raise ValueError("The evaluation loader is empty")
    return {"loss": total_loss / count, "accuracy": correct / count, "correct": correct,
            "samples": count, "predictions": predictions, "targets": targets}


def plot_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    """Writing the two training graphs for the report."""
    import matplotlib
    # Agg draws directly to image files and works without a display window.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    for metric, label, filename, color in [
        ("loss", "Training cross-entropy loss", "training_loss.png", "#205b90"),
        ("accuracy", "Training accuracy (%)", "training_accuracy.png", "#17664b"),
    ]:
        values = [row[metric] * (100 if metric == "accuracy" else 1) for row in history]
        fig, ax = plt.subplots(figsize=(6.6, 3.5), layout="constrained")
        ax.plot(epochs, values, marker="o", color=color, linewidth=2)
        ax.set(xlabel="Epoch", ylabel=label, xticks=epochs)
        ax.grid(alpha=0.25)
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def run_training(config: TrainConfig) -> dict[str, Any]:
    """Running the complete program and saving then measurements and final model."""
    config.validate()
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Preserving an earlier experiment instead of overwriting its evidence.
    if (out / "results.json").exists() or (out / "checkpoint_last.pt").exists():
        raise FileExistsError(f"Use a fresh output directory; {out} already contains a run")
    torch.set_num_threads(config.cpu_threads)
    # Seeding before initializing weights and the loader's shuffle generator.
    seed_everything(config.seed)
    device = choose_device(config.device)
    model = ThreeLayerCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # The two bonus functions inspect the model; they do not train it.
    parameters = count_parameters(model)
    flops = calculate_flops(model)
    assert parameters["total"] == 33_834
    assert flops["total_flops"] == 2_811_904
    train_loader, test_loader = make_loaders(config)
    training_dataset = train_loader.dataset
    testing_dataset = test_loader.dataset
    # We need to check that it is actually a CIFAR-10 dataset before accessing its specific attributes.
    # CIFAR10 type before using its __len__() and classes attributes.
    if not isinstance(training_dataset, datasets.CIFAR10) or not isinstance(testing_dataset, datasets.CIFAR10):
        raise TypeError("Expected CIFAR-10 training and test datasets")
    training_samples = len(training_dataset)
    test_samples = len(testing_dataset)
    assert training_samples == 50_000 and test_samples == 10_000
    print(f"\nDevice: {device}; train samples: 50,000; test samples: 10,000", flush=True)
    history = []
    started = time.perf_counter()
    # Records each epoch immediately so progress survives an interrupted run.
    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "accuracy", "correct", "samples", "seconds"])
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            print(f"Epoch {epoch}/{config.epochs}", flush=True)
            epoch_started = time.perf_counter()
            metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            row = {"epoch": epoch, **metrics, "seconds": time.perf_counter() - epoch_started}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(f"  loss={row['loss']:.6f}; accuracy={100*row['accuracy']:.2f}%; "
                  f"samples={row['samples']:,}; seconds={row['seconds']:.1f}", flush=True)
            # Save every completed epoch; the final file holds epoch 10.
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "config": asdict(config),
                        "history": history}, out / "checkpoint_last.pt")
    training_seconds = time.perf_counter() - started
    # The test set is touched only after all weight updates are complete.
    print("Evaluating the final model on the full test split...", flush=True)
    test_result = evaluate(model, test_loader, criterion, device)
    predictions = test_result.pop("predictions")
    targets = test_result.pop("targets")
    with (out / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["test_index", "target", "prediction", "correct"])
        # Testing is not shuffled, so row i still corresponds to test image i.
        writer.writerows((i, y, p, int(y == p)) for i, (y, p) in enumerate(zip(targets, predictions)))
    plot_history(history, out)
    import torchvision
    result = {
        "config": asdict(config), "device": str(device), "training_seconds": training_seconds,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "platform": platform.platform()},
        "dataset": {"name": "CIFAR-10", "training_samples": training_samples,
                    "test_samples": test_samples, "classes": training_dataset.classes},
        "parameters": parameters, "flops": flops, "history": history, "test": test_result,
        # Fingerprint the exact source that produced these recorded results.
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Final test loss={test_result['loss']:.6f}; accuracy={100*test_result['accuracy']:.2f}% "
          f"({test_result['correct']:,}/{test_result['samples']:,})", flush=True)
    return result


def main() -> None:
    """Parse command-line options and run training or only bonus calculations."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [("epochs", 10), ("batch-size", 128), ("seed", 725), ("num-workers", 0), ("cpu-threads", 4)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--complexity-only", action="store_true")
    args = parser.parse_args()
    if args.complexity_only:
        # This quick mode needs neither a dataset download nor ten epochs.
        model = ThreeLayerCNN()
        count_parameters(model)
        calculate_flops(model)
    else:
        options = vars(args)
        options.pop("complexity_only")
        run_training(TrainConfig(**options))


if __name__ == "__main__":
    main()
