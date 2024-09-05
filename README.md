# Adversarial attacks for machine learning denoisers and how to resist them

This repository implements a deep learning approach for **denoising and enhancing cell segmentation of microscopic images**. The project focuses on using a StarDist-based cell segmentation model, incorporating polynomial denoising technique, and leveraging optimization algorithms to improve segmentation results on noisy microscopic images. 

This work is part of a research paper published at SPIE (https://doi.org/10.1117/12.2632954).

## Project Overview

### Objective:
The goal is to improve the cell segmentation of [DSB 2018 nuclei segmentation challenge dataset](https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip) by reducing noise through polynomial optimization. 
The project utilizes:
- **[StarDist](https://github.com/stardist/stardist)** model for cell detection and segmentation.
- **[MINTERPY](https://github.com/casus/minterpy)** framework for polynomial noise generation.
- **Denoising and Optimization**: Leverages `Hyppopy` for hyperparameter optimization to fine-tune denoising parameters for the best segmentation performance.
- **False Positive/Negative Detection**: Post-processes segmentation results to reduce False Positives and False Negatives through various noise models.
- **Optimization of false positives (FP) and false negatives (FN)** in the segmentation results.

The **Results** contain the artifacts of potential False Positives and potential False Negative segemntation detected using the proposed optimization. The results are also compared to that of Gaussian and Poisson noises qualitatively and quantitatively.