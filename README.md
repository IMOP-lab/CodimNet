# CodimNet

![](figures/CodimNet.jpg)

Figure 1: Detailed network structure of our proposed CodimNet.

## Methods

### Multibranch

![](figures/Multibranch.jpg)

Figure 2: Visualize the multi-branch structure.

### ND-PINN

![](figures/ND-PINN.jpg)

Figure 3: Visualize the ND-PINN structure.

## Installation

We run CodimNet and previous methods on a system running Ubuntu 22.04, with Python 3.8, PyTorch 2.1.0, and CUDA 12.1.

## Experiment

### Models Evaluation Without Cross-Validation

![](tables/Hold-Out%20Validation.jpg)

Figure 4: Comparison of CodimNet and other methods for AD diagnosis on the CAUEEG dataset using a fixed hold-out validation approach with EEG data.

### Models Evaluation Using Cross-Validation

![](tables/Four-Fold%20Cross-Validation.jpg)

Figure 5: Comparison of CodimNet and other methods for AD diagnosis on the non-overlapping version of the CAUEEG dataset using a four-fold cross-validation approach with EEG data.
