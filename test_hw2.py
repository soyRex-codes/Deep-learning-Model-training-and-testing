"""Independent tests for architecture, manual complexity, and training metrics."""
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import hw2_solution as solution


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(4)
    torch.manual_seed(725)


def test_exact_required_architecture():
    model = solution.ThreeLayerCNN()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert [(m.in_channels, m.out_channels) for m in convs] == [(3, 16), (16, 32), (32, 64)]
    assert all(m.kernel_size == (3, 3) and m.stride == (1, 1) and m.padding == (1, 1)
               and m.bias is not None for m in convs)
    assert len([m for m in model.modules() if isinstance(m, nn.ReLU)]) == 3
    pools = [m for m in model.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(pools) == 3
    assert all(m.kernel_size == 2 and m.stride == 2 for m in pools)
    assert model.fc.in_features == 1024 and model.fc.out_features == 10
    assert not any(isinstance(m, (nn.Softmax, nn.BatchNorm2d, nn.Dropout)) for m in model.modules())


@pytest.mark.parametrize("batch_size", [1, 7])
def test_output_and_pool_shapes(batch_size):
    model = solution.ThreeLayerCNN()
    observed = []
    handles = [m.register_forward_hook(lambda layer, args, output: observed.append(tuple(output.shape)))
               for m in [model.pool1, model.pool2, model.pool3]]
    try:
        logits = model(torch.randn(batch_size, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert logits.shape == (batch_size, 10)
    assert observed == [(batch_size, 16, 16, 16), (batch_size, 32, 8, 8), (batch_size, 64, 4, 4)]
    assert torch.isfinite(logits).all()


def test_parameters_match_independent_manual_counts():
    model = solution.ThreeLayerCNN()
    result = solution.count_parameters(model, verbose=False)
    manual = [16 * (3 * 3 * 3 + 1), 32 * (16 * 3 * 3 + 1),
              64 * (32 * 3 * 3 + 1), 10 * (1024 + 1)]
    assert [r["total"] for r in result["layers"]] == manual == [448, 4640, 18496, 10250]
    assert result["total"] == sum(manual) == 33834
    assert result["total"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert [r["biases"] for r in result["layers"]] == [16, 32, 64, 10]


def test_frozen_parameters_are_not_learnable():
    model = solution.ThreeLayerCNN()
    for p in model.conv1.parameters():
        p.requires_grad_(False)
    assert solution.count_parameters(model, verbose=False)["total"] == 33834 - 448


@pytest.mark.parametrize("input_size", [(3, 32, 32), (1, 3, 32, 32), (5, 3, 32, 32)])
def test_flops_match_manual_values_and_batch_scaling(input_size):
    result = solution.calculate_flops(solution.ThreeLayerCNN(), input_size, verbose=False)
    manual = [32 * 32 * 16 * 3 * 9, 16 * 16 * 32 * 16 * 9,
              8 * 8 * 64 * 32 * 9, 1024 * 10]
    assert [r["flops"] for r in result["layers"]] == manual == [442368, 1179648, 1179648, 10240]
    assert result["total_flops"] == sum(manual) == 2811904
    n = input_size[0] if len(input_size) == 4 else 1
    assert result["batch_flops"] == n * 2811904


def test_grouped_convolution_and_biasless_linear():
    model = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=False),
                          nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(96, 3, bias=False))
    p = solution.count_parameters(model, verbose=False)
    f = solution.calculate_flops(model, (2, 4, 8, 8), verbose=False)
    assert p["total"] == 6 * 2 * 9 + 96 * 3 == 396
    assert f["total_flops"] == 8 * 8 * 6 * 2 * 9 + 96 * 3 == 7200
    assert f["batch_flops"] == 14400


def test_complexity_calculation_preserves_modes_parameters_and_hooks():
    model = solution.ThreeLayerCNN().train()
    model.pool1.eval()
    modes = [m.training for m in model.modules()]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    solution.calculate_flops(model, verbose=False)
    assert [m.training for m in model.modules()] == modes
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
    with pytest.raises(RuntimeError):
        solution.calculate_flops(model, (1, 4, 32, 32), verbose=False)
    assert [m.training for m in model.modules()] == modes
    assert all(len(m._forward_hooks) == 0 for m in model.modules())


@pytest.mark.parametrize("bad", [(0, 3, 32, 32), (3, 32), (1, 3.0, 32, 32), (-1, 3, 32, 32)])
def test_flops_reject_invalid_dimensions(bad):
    with pytest.raises(ValueError):
        solution.calculate_flops(solution.ThreeLayerCNN(), bad, verbose=False)


def test_evaluation_weights_short_final_batch_correctly():
    logits = torch.tensor([[5., 0.], [0., 5.], [5., 0.], [0., 5.], [0., 5.]])
    labels = torch.tensor([0, 1, 0, 1, 0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=3)
    model = nn.Identity().train()
    result = solution.evaluate(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert result["loss"] == pytest.approx(nn.CrossEntropyLoss()(logits, labels).item(), rel=1e-6)
    assert result["correct"] == 4 and result["samples"] == 5 and result["accuracy"] == 0.8
    assert result["predictions"] == [0, 1, 0, 1, 1]
    assert not model.training


def test_training_computes_gradients_and_updates_weights():
    model = solution.ThreeLayerCNN().eval()
    images = torch.randn(5, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3, 4])
    before = model.fc.weight.detach().clone()
    loader = DataLoader(TensorDataset(images, labels), batch_size=3, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    result = solution.train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu"))
    assert model.training and result["samples"] == 5
    assert math.isfinite(result["loss"]) and 0 <= result["accuracy"] <= 1
    assert not torch.equal(before, model.fc.weight)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_dataset_splits_and_shuffle_setting(monkeypatch, tmp_path):
    calls = []
    def fake_cifar10(**kwargs):
        calls.append(kwargs)
        return TensorDataset(torch.zeros(5, 3, 32, 32), torch.zeros(5, dtype=torch.long))
    monkeypatch.setattr(solution.datasets, "CIFAR10", fake_cifar10)
    train, test = solution.make_loaders(solution.TrainConfig(data_dir=str(tmp_path)))
    assert [c["train"] for c in calls] == [True, False]
    assert all(c["download"] for c in calls)
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(test.sampler, SequentialSampler)
    assert not train.drop_last and not test.drop_last


def test_assignment_minimum_epochs_enforced():
    with pytest.raises(ValueError, match="at least 10"):
        solution.TrainConfig(epochs=9).validate()
    solution.TrainConfig(epochs=10).validate()
