# Machine Learning for IBD-associated Fatigue

## Overview

This repository contains the data and code used to create machine learning models for studying IBD-associated fatigue. The notebooks implement deep neural networks and other ML approaches, with analyses performed on both the full IBD cohort and a subset of patients in biochemical remission (defined as being asymptomatic + faecal calprotectin <250 ug/g + C-reactive protein <5 mg/L).

## Pipeline Summary

![ML Pipeline Summary](pipeline_summary.png)

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
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/1-gut/machine-learning-for-ibd-fatigue.git
```

### 2. Set up a Python virtual environment

```bash
cd machine-learning-for-ibd-fatigue
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
├── output/                        # Model outputs and visualizations
├── src/                          # Source code modules
├── tests/                        # Test files
├── archived_notebooks/           # Previous notebook versions
├── 1_demographics.ipynb          # Cohort descriptive statistics
├── 2_dnn.ipynb                   # Deep neural network implementation
├── 3_dnn_biochemical_remission.ipynb  # DNN for remission cohort
├── 5_ml_pipeline_all_ibd.ipynb   # Full ML pipeline for all patients
├── 6_ml_pipeline_biochem_remission.ipynb  # ML pipeline for remission cohort
├── 8_validation_spain.ipynb      # External validation on Spanish cohort
├── 9_traditional_logreg_using_statsmodel.ipynb  # Traditional statistics
└── 10_validation_scotland.ipynb  # External validation on Scottish cohort
```

## Usage

We recommend using VSCode to run the Jupyter notebooks. Select the virtual environment created above as your Python interpreter.

Note on units: Blood test measurements use different units between cohorts:

- Scotland: Urea (mmol/L), Creatinine (umol/L)
- Spain: Urea (mg/dL), Creatinine (mg/dL)

## Citation

If you use code or models from this repository, please cite the associated paper [citation details to be added].

## Feedback

For questions or feedback, please contact:

- Shaun Chuah ([shaun.chuah@glasgow.ac.uk](mailto:shaun.chuah@glasgow.ac.uk))

## Repository Authors

- Shaun Chuah ([github.com/shaunchuah](https://github.com/shaunchuah))
- Robert Whelan ([github.com/rw509](https://github.com/rw509))

## Data Contributors

[to be added]
