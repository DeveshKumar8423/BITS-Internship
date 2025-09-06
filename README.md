# BITS-Internship: Cross-Domain Semantic Segmentation

## Overview

This project implements cross-domain semantic segmentation using deep learning models such as **DeepLabV3+** and **DaFormer**. It provides a modular pipeline for training, evaluation, visualization, and pseudo-label generation on datasets like **Cityscapes** and **IDD**.

## Folder Structure

cross_domain_segmentation/
├── src/               # Source code (models, data loading, training, evaluation, utils)
├── visualizations/    # Visualization outputs (comparisons, sample predictions)
├── configs/           # Configuration files for experiments
├── scripts/           # Shell scripts for quick training/evaluation
├── main.py            # Main entry point for running experiments
├── requirements.txt   # Python dependencies

## Features

* **Modular Data Loading**: Supports Cityscapes and IDD datasets with configurable transforms.
* **Model Support**: DeepLabV3+ and DaFormer architectures.
* **Training & Adaptation**: Baseline and domain adaptation training routines.
* **Evaluation**: Quantitative and qualitative evaluation, including confusion matrix and visualizations.
* **Pseudo-Label Generation**: Save model predictions as pseudo-labels for semi-supervised learning.
* **Visualization**: Compare model outputs and generate visual summaries.

## Installation

1.  **Clone the repository:**
    git clone [https://github.com/DeveshKumar8423/BITS-Internship.git](https://github.com/DeveshKumar8423/BITS-Internship.git)
    cd BITS-Internship/cross_domain_segmentation

2.  **Set up a Python environment:**
    python3 -m venv .venv
    source .venv/bin/activate

3.  **Install dependencies:**
    pip install -r requirements.txt
    

---

## Usage

### Training

**Baseline Training:**
python main.py --config configs/cityscapes_config.py --mode train_baseline

**Domain Adaptation Training**
python main.py --config configs/idd_config.py --mode train_adapt

**Evaluate Model**
python main.py --config configs/cityscapes_config.py --mode evaluate --checkpoint <path_to_checkpoint>

**Visualize Comparison**
python visualize_comparison.py --config configs/cityscapes_config.py --checkpoint <path_to_checkpoint>

## Configuration
Configuration files are located in configs/ and define dataset paths, model parameters, training settings, etc.

Examples: cityscapes_config.py, idd_config.py

## Results
Visual results and metrics are saved in visualizations/ and logs/.

Pseudo-labels are saved in pseudo_labels/ (if enabled).

## Scripts
Quick scripts for training and evaluation are available in the scripts/ folder:

quick_train_baseline.sh

quick_train_adapt.sh

quick_evaluate.sh

quick_refine_labels.sh

## Contributing
Feel free to open issues or pull requests for improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License.

## Contact
For questions or collaboration, contact DeveshKumar8423.