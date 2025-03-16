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

## Ablation study

### Branch Ablation Study

![](tables/Ablation%20study%20of%20Partitioned.jpg)

Figure 6: Quantitative assessment of regional contributions to AD classification efficacy through cortical branch-specific ablation analysis.

### ND-PINN Quantitative Ablation Study

![](tables/Ablation%20study%20of%20Quantification.jpg)

Figure 7: Quantitative evaluation of model classification efficacy across AD EEG tri-classification under varying instantiations of ND-PINN.

### Ablation Study Of ND-PINN In Different Intermediate Layers

![](tables/Ablation%20study%20of%20Layers.jpg)

Figure 8: Quantitative evaluation of ND-PINN instantiation across distinct intermediate-layer constraints. 

### Ablation Study Of Different Branches Of ND-PINN

![](tables/Ablation%20study%20of%20Removed.jpg)

Figure 9: Quantitative assessment of the impact of ND-PINN removal across distinct cortical branches on classification performance. 

### Branch Different Model Ablation Study

![](tables/Ablation%20study%20of%20Network.jpg)

Figure 10: Quantitative classification performance across distinct recurrent neural network architectures instantiated within a multibranch paradigm. 

### ND-PINN constrained different brain rhythm Ablation Study

![](tables/Ablation%20study%20of%20Rhythmicity.jpg)

Figure 11: Performance metrics of ND-PINN under differential brain rhythm constraints, evaluating the impact of rhythm-specific spectral ablation on AD EEG classification.  






