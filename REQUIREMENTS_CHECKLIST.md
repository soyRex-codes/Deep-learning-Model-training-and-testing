# HW2 requirement-by-requirement verification

| Requirement | Status | Evidence |
|---|---|---|
| Setup - torchvision CIFAR10 splits | PASS | hw2_solution.py: make_loaders; full 50,000/10,000 verified |
| Setup - DataLoader mini-batches and training shuffle | PASS | make_loaders; RandomSampler/SequentialSampler tested |
| Architecture - three exact convolution blocks | PASS | ThreeLayerCNN; exact settings and layer counts tested |
| Architecture - flatten and ten-class output | PASS | 64x4x4=1,024; output [N,10] tested |
| Training - cross-entropy and Adam | PASS | run_training and train_one_epoch |
| Training - at least ten complete epochs | PASS | 10 epochs; 50,000 examples every epoch |
| Training - record loss and accuracy per epoch | PASS | results/history.csv and results/results.json |
| Testing - final full test evaluation | PASS | 6,995/10,000 correct; 69.95% |
| Report - architecture, methods, epoch table, graphs | PASS | HW2_Report.pdf / HW2_Report.md |
| Report - layer roles, results, implementation challenges | PASS | Report sections 2, 4 and 7 |
| Bonus - custom parameter count with weights and biases | PASS | Per-layer counts; total 33,834 |
| Bonus - custom FLOPs using actual convolution outputs | PASS | Per-layer counts; total 2,811,904 per image |
| Bonus - assignment-specific counting convention | PASS | No addition, bias, ReLU or pooling cost; no profiler library |
| Bonus - explanations and manual verification | PASS | Conv1 and linear hand checks in report agree with code |
| Verification - saved model and all test predictions | PASS | results/verification.json; checkpoint retest reproduces all predictions |

These checks establish compliance with the listed assignment requirements. Instructor grading remains separate.
