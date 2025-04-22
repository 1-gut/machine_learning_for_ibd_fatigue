# Machine Learning for IBD-associated Fatigue

## Overview

This repository contains the code and implementation of machine learning models for predicting and studying Inflammatory Bowel Disease (IBD) associated fatigue. The work includes deep neural networks and traditional machine learning approaches, with analyses performed on both the full IBD cohort and a subset of patients in biochemical remission (defined as being asymptomatic + faecal calprotectin <250 μg/g + C-reactive protein <5 mg/L).

## Pipeline Summary

![ML Pipeline Summary](pipeline_summary.png)

## Key Features

- Deep neural network models using TensorFlow and PyTorch
- Traditional machine learning models (Random Forest, XGBoost, SVM, etc.)
- Explainable AI with SHAP (SHapley Additive exPlanations)
- External validation on independent cohorts from Australi, Spain and Scotland
- Benchmarking ML models against traditional statistical methods
- Comprehensive evaluation metrics and visualizations

## Requirements

- Python 3.11.9 [Install Here](https://www.python.org/downloads/)
- Git [Install Here](https://git-scm.com/downloads)

Key dependencies:

```text
tensorflow==2.18.0
torch==2.5.1
scikit-learn==1.5.2
pandas==2.2.3
numpy==2.0.2
shap==0.46.0
seaborn==0.13.2
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/1-gut/machine-learning-for-ibd-fatigue.git
cd machine-learning-for-ibd-fatigue
```

### 2. Set up a Python virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For MacOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
.\venv\Scripts\activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

## Directory Structure

```bash
machine-learning-for-ibd-fatigue/
├── data/                          # Anonymised data
│   └── ...                        # Various data files
├── output/                        # Model outputs and visualizations
│   ├── demographics/              # Cohort statistics outputs
│   ├── all_ibd/                   # Results for full IBD cohort
│   └── biochemical_remission/     # Results for remission cohort
├── src/                           # Source code modules
│   ├── config/                    # Configuration files
│   ├── data/                      # Data processing utilities
│   ├── models/                    # Model definitions
│   └── visualization/             # Plotting and visualization tools
├── archived_notebooks/            # Previous notebook versions, contains reference PyTorch implementation
├── 1_demographics.ipynb           # Cohort descriptive statistics
├── 2_dnn.ipynb                    # Deep neural network implementation
├── 3_dnn_biochemical_remission.ipynb  # DNN for remission cohort
├── 5_ml_pipeline_all_ibd.ipynb    # Full ML pipeline for all patients
├── 6_ml_pipeline_biochem_remission.ipynb  # ML pipeline for remission cohort
├── 8_validation_spain.ipynb       # External validation on Spanish cohort
├── 9_traditional_logreg_using_statsmodel.ipynb  # Traditional statistics
├── 10_validation_scotland.ipynb   # External validation on Scottish cohort
├── requirements.txt               # Package dependencies
└── LICENSE                        # Apache License 2.0
```

## Notebook Descriptions

| Notebook | Description |
|----------|-------------|
| 1_demographics.ipynb | Analysis of cohort characteristics and statistical comparisons |
| 2_dnn.ipynb | Implementation of deep neural network for fatigue prediction |
| 3_dnn_biochemical_remission.ipynb | DNN model specifically for patients in biochemical remission |
| 5_ml_pipeline_all_ibd.ipynb | Complete ML pipeline comparing multiple algorithms on full cohort |
| 6_ml_pipeline_biochem_remission.ipynb | ML pipeline for biochemical remission cohort |
| 8_validation_spain.ipynb | External validation using Spanish cohort data |
| 9_traditional_logreg_using_statsmodel.ipynb | Traditional statistical analysis using logistic regression |
| 10_validation_scotland.ipynb | External validation using Scottish cohort data |

## Usage

### Running the Notebooks

We recommend using VSCode to run the Jupyter notebooks. Select the virtual environment created above as your Python interpreter.

```bash
code .  # Open project in VSCode
```

Alternatively, start Jupyter lab or notebook server:

```bash
jupyter lab
# or
jupyter notebook
```

### Note on Units

Blood test measurements use different units between cohorts:

- Scotland: Urea (mmol/L), Creatinine (μmol/L)
- Spain: Urea (mg/dL), Creatinine (mg/dL)

Conversion functions are included in the codebase to standardize these measurements.

## Output Directory

The output directory contains the results of the models and visualizations:

```bash
├── output
│   ├── demographics                  # Cohort statistics outputs
│   ├── all_ibd                       # Results for full IBD cohort
│   │   ├── dnn                       # Deep neural network outputs
│   │   │   ├── exports               # Exported models and data
│   │   │   ├── plots                 # Visualization plots
│   │   │   │   └── forceplots        # SHAP force plots
│   │   │   ├── metrics               # Model performance metrics
│   │   │   └── evaluation            # Model evaluation results
│   │   ├── sklearn                   # Traditional ML model outputs
│   │   │   ├── exports               # Exported models and data
│   │   │   ├── plots                 # Visualization plots
│   │   │   ├── metrics               # Model performance metrics
│   │   │   └── evaluation            # Model evaluation results
│   │   └── validation                # External validation results
│   │       ├── spain                 # Spanish cohort validation
│   │       └── scotland              # Scottish cohort validation
│   └── biochemical_remission         # Results for remission cohort
│       └── ...                       # Similar structure as all_ibd
```

## Reproducing Our Results

To reproduce our results:

1. Run the notebooks in numerical order (1_demographics.ipynb → 2_dnn.ipynb → etc.)
2. Each notebook saves outputs to the appropriate directory in output
3. Validation notebooks (8, 10) rely on trained models from earlier notebooks

## Citation

If you use code or models from this repository in your research, please cite:

```
[citation details to be added after publication]
```

## Feedback and Contributions

For questions, feedback, or contributions, please:

- Open an issue on GitHub
- Submit a pull request with proposed changes
- Contact the authors directly

## Contact

- Shaun Chuah ([shaun.chuah@glasgow.ac.uk](mailto:shaun.chuah@glasgow.ac.uk))
- [Gut Translational Research Group Lab Website](https://www.gla.ac.uk/schools/infectionimmunity/staff/gwotzerho/theguttranslationalresearchgroup/)

## Repository Authors

- Shaun Chuah ([github.com/shaunchuah](https://github.com/shaunchuah))
- Robert Whelan ([github.com/rw509](https://github.com/rw509))

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
