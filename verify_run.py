"""Verify completed HW2 artifacts and optionally re-evaluate the saved model."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from hw2_solution import ThreeLayerCNN, TrainConfig, choose_device, count_parameters, calculate_flops, evaluate, make_loaders


def executable_source_hash(source: str) -> str:
    """Hash Python statements while ignoring comments and docstrings."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    normalized = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def verify(result_dir: Path, *, reload_test: bool = False) -> dict:
    results = json.loads((result_dir / "results.json").read_text())
    expected_parameters = {"conv1": 448, "conv2": 4640, "conv3": 18496, "fc": 10250}
    expected_flops = {"conv1": 442368, "conv2": 1179648, "conv3": 1179648, "fc": 10240}
    config = TrainConfig(**results["config"])
    config.validate()
    history = results["history"]
    assert len(history) == config.epochs >= 10
    assert [r["epoch"] for r in history] == list(range(1, config.epochs + 1))
    assert results["dataset"]["training_samples"] == 50000
    assert results["dataset"]["test_samples"] == 10000
    for row in history:
        assert row["samples"] == 50000
        assert math.isfinite(row["loss"]) and row["loss"] >= 0
        assert row["accuracy"] == row["correct"] / row["samples"]
        assert 0 <= row["accuracy"] <= 1
    assert results["test"]["samples"] == 10000
    assert results["test"]["accuracy"] == results["test"]["correct"] / 10000
    assert {r["layer"]: r["total"] for r in results["parameters"]["layers"]} == expected_parameters
    assert {r["layer"]: r["flops"] for r in results["flops"]["layers"]} == expected_flops
    assert results["parameters"]["total"] == 33834
    assert results["flops"]["total_flops"] == 2811904
    with (result_dir / "history.csv").open() as stream:
        csv_history = list(csv.DictReader(stream))
    assert len(csv_history) == len(history)
    for a, b in zip(csv_history, history):
        assert int(a["epoch"]) == b["epoch"]
        assert float(a["loss"]) == b["loss"]
        assert float(a["accuracy"]) == b["accuracy"]
    with (result_dir / "test_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == 10000
    assert [int(r["test_index"]) for r in predictions] == list(range(10000))
    assert all(0 <= int(r["target"]) < 10 and 0 <= int(r["prediction"]) < 10 for r in predictions)
    assert sum(int(r["correct"]) for r in predictions) == results["test"]["correct"]
    assert all(int(r["correct"]) == int(r["target"] == r["prediction"]) for r in predictions)
    for filename in ["training_loss.png", "training_accuracy.png", "checkpoint_last.pt"]:
        assert (result_dir / filename).stat().st_size > 0
    source = Path(__file__).with_name("hw2_solution.py")
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() == results["source_sha256"]:
        source_check = "Training source SHA-256 matches saved run"
    else:
        # Comments, docstrings, and formatting can change after training without
        # changing executable statements. Compare those statements in that case.
        expected_ast_hash = results.get("source_ast_sha256")
        assert expected_ast_hash is not None, "Training source changed without a recorded syntax fingerprint"
        assert executable_source_hash(source_bytes.decode("utf-8")) == expected_ast_hash
        source_check = "Current executable statements match the recorded training source"

    checkpoint = torch.load(result_dir / "checkpoint_last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == config.epochs
    model = ThreeLayerCNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert count_parameters(model, verbose=False)["total"] == 33834
    assert calculate_flops(model, verbose=False)["total_flops"] == 2811904
    verification = {
        "status": "PASS", "completed_epochs": len(history), "training_samples_per_epoch": 50000,
        "final_test_samples": 10000, "parameters": 33834, "flops_per_image": 2811904,
        "checks": ["Exact architecture and state-dict load", "10+ full training epochs",
                   "CSV and JSON metrics agree", "All 10,000 test predictions accounted for",
                   "Per-layer and total complexity match independent manual values",
                   "Training graphs exist", source_check],
        "saved_model_retested": False,
    }
    if reload_test:
        torch.set_num_threads(config.cpu_threads)
        device = choose_device(results["device"])
        model.to(device)
        _, test_loader = make_loaders(config)
        retest = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        assert retest["correct"] == results["test"]["correct"]
        assert abs(retest["loss"] - results["test"]["loss"]) < 1e-6
        assert retest["predictions"] == [int(r["prediction"]) for r in predictions]
        verification["saved_model_retested"] = True
        verification["checks"].append("Reloaded checkpoint reproduces every test prediction and final metrics")
    (result_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--reload-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.results, reload_test=args.reload_test), indent=2))
