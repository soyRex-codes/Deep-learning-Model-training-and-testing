# CSC725 Homework 2 - complete implementation and bonus analysis

This package implements the exact three-layer CNN in the assignment. It includes
the full CIFAR-10 training/evaluation workflow, hand-written complexity functions,
tests, recorded results, training graphs, and the report.

## Run it

Use Python 3.12 or another Python version supported by the listed PyTorch versions.
Create an isolated environment rather than installing into a system environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q test_hw2.py
python hw2_solution.py --epochs 10 --data-dir data --output-dir results_new
python verify_run.py --results results_new --reload-test
```

Run these commands from the folder containing this README. A new output folder
prevents an accidental overwrite of the supplied experiment. The first run needs
internet access to download CIFAR-10; later runs reuse the checked dataset.
The device is selected automatically (CUDA, then Apple MPS, then CPU). Use
`--device cpu` to reproduce the supplied CPU run as closely as possible. Fixed
seeds improve repeatability; results can still differ across software/hardware.

`HW2_Walkthrough.ipynb` is an optional notebook that explains each stage and
loads the completed experiment by default. Its cells were not used to produce
the recorded run; `hw2_solution.py` was. To use it in Jupyter or VS Code, install
`ipykernel` in this same environment and select that environment as its kernel.

For only the bonus calculations:

```bash
python hw2_solution.py --complexity-only
```

## Step 1 - load the correct data

`torchvision.datasets.CIFAR10(train=True)` loads all 50,000 training images.
`train=False` loads the separate 10,000 test images. `ToTensor()` produces a
3 x 32 x 32 RGB tensor; normalization maps each channel from [0,1] to [-1,1].
DataLoader makes batches of 128. Training uses `shuffle=True`; testing uses
`shuffle=False`. `drop_last=False` keeps every image, including the short final
batch. No augmentation, validation search, or tuning on the test set is used.

## Step 2 - follow the specified architecture

| Operation | Shape for one image |
|---|---|
| Input | 3 x 32 x 32 |
| Conv1 (3 -> 16), ReLU | 16 x 32 x 32 |
| MaxPool1 | 16 x 16 x 16 |
| Conv2 (16 -> 32), ReLU | 32 x 16 x 16 |
| MaxPool2 | 32 x 8 x 8 |
| Conv3 (32 -> 64), ReLU | 64 x 8 x 8 |
| MaxPool3 | 64 x 4 x 4 |
| Flatten | 1,024 |
| Fully connected | 10 raw class logits |

Every convolution has kernel 3 x 3, stride 1, padding 1, and a bias. Every pool
has kernel 2 x 2 and stride 2. There are no extra convolutions, normalization
layers, dropout layers, or additional classifier layers.

The shape formula is `floor((input + 2*padding - kernel)/stride) + 1`.
For a 3 x 3/S1/P1 convolution, a spatial dimension stays the same. Pooling halves
32 to 16 to 8 to 4. This is why the linear layer needs 64*4*4 = 1,024 inputs.

## Step 3 - train for at least ten epochs

The training loop uses cross-entropy and Adam with learning rate 0.001. For each
batch, it clears old gradients, obtains logits, computes loss, backpropagates,
and updates the weights. `model.train()` selects training mode.

There is no softmax layer before cross-entropy: PyTorch's loss expects logits.
For each epoch, loss is the sample-weighted mean of the batch losses and accuracy
is total correct/total examples. These are **online training metrics**: each batch
is measured before its optimizer update, so they are not a separate evaluation
of the final epoch model. The code logs each epoch and saves a checkpoint.

## Step 4 - evaluate only after training

`model.eval()` selects evaluation mode and `torch.no_grad()` disables gradient
tracking. The final checkpoint is evaluated on every test image. The output
includes test accuracy, test loss, and all individual test predictions. The
verification script reloads that checkpoint and checks the predictions and
metrics again; this is a reproducibility check, not test-set model selection.

## Step 5 - solve the entire bonus section

`count_parameters(model)` counts trainable weight and bias tensor elements
directly, reports each layer, and returns the total. Shared parameters are counted
once; frozen parameters are not learnable and are excluded.

`calculate_flops(model, input_size)` observes actual output shapes using temporary
forward hooks, then applies the assignment formulas itself. It restores all
training modes and removes hooks after the calculation. It uses no profiling
package. It reports counts per image and separately for a supplied batch.

| Layer | Parameters (weights + biases) | Multiplications per image |
|---|---:|---:|
| Conv1 | 16*(3*3*3+1) = 448 | 32*32*16*3*3*3 = 442,368 |
| Conv2 | 32*(16*3*3+1) = 4,640 | 16*16*32*16*3*3 = 1,179,648 |
| Conv3 | 64*(32*3*3+1) = 18,496 | 8*8*64*32*3*3 = 1,179,648 |
| Linear | 10*(1,024+1) = 10,250 | 1,024*10 = 10,240 |
| **Total** | **33,834** | **2,811,904** |

One multiplication counts as one FLOP in this homework. Additions, bias
operations, ReLU, and max-pooling are excluded. The count is not a wall-clock
performance measurement and does not include backward propagation or Adam.

## Step 6 - review the evidence and report

- `results/history.csv`: recorded loss, accuracy, sample counts, and time per epoch.
- `results/results.json`: configuration, environment, all metrics and complexity.
- `results/training_loss.png`, `training_accuracy.png`: required graphs.
- `results/test_predictions.csv`: all test predictions and correct/incorrect flags.
- `results/checkpoint_last.pt`: the trained model and optimizer states.
- `results/verification.json`: checks on the completed experiment and saved model.
- `test_hw2.py`: independent tests of architecture, bonus formulas, gradients,
  sample-weighted metrics, data split/shuffle settings, and mode restoration.
- `HW2_Report.pdf`: the report with actual experimental measurements.

Read the report and walkthrough so you can explain the implementation. The PDF
assignment defines the grading requirements; meeting them does not guarantee a
particular instructor score.

## Official API references

- [CIFAR10 dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [DataLoader](https://docs.pytorch.org/docs/stable/data.html)
# Deep-learning
