# Machine Learning for IBD-associated Fatigue

## 🌟 Overview

This repository contains the code and implementation of machine learning models for predicting and studying Inflammatory Bowel Disease (IBD) associated fatigue. The work includes deep neural networks and traditional machine learning approaches, with analyses performed on both the full IBD cohort and a subset of patients in biochemical remission (defined as being asymptomatic + faecal calprotectin <250 μg/g + C-reactive protein <5 mg/L).

## 🛠️ Pipeline Summary

![ML Pipeline Summary](pipeline_summary.png)

## ✨ Key Features

- Deep neural network models using TensorFlow and PyTorch
- Traditional machine learning models (Random Forest, XGBoost, SVM, etc.)
- Explainable AI with SHAP (SHapley Additive exPlanations)
- External validation on independent cohorts from Australia, Spain and Scotland
- Benchmarking ML models against traditional statistical methods
- Comprehensive evaluation metrics and visualizations

## 📋 Requirements

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
matplotlib==3.9.2
statsmodels==0.15.0
```

## 💻 Installation

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

## 📁 Directory Structure

```bash
machine-learning-for-ibd-fatigue/
├── data/                          # Anonymised data
│   └── cohort_online_cucq32.csv   # Main cohort data
├── output/                        # Model outputs and visualizations
│   ├── demographics/              # Cohort statistics outputs
│   ├── all_ibd/                   # Results for full IBD cohort
│   ├── biochemical_remission/     # Results for full IBD cohort
│   └── validation/                # Validation on external cohorts
├── src/                           # Source code modules
│   ├── config/                    # Configuration files and constants
│   ├── data/                      # Data processing utilities
│   ├── models/                    # Model definitions
│   └── visualization/             # Plotting and visualization tools
├── archived_notebooks/            # Previous notebooks
├── 1_demographics.ipynb           # Cohort descriptive statistics
├── 2_dnn.ipynb                    # Deep neural network implementation
├── 3_dnn_biochemical_remission.ipynb  # DNN for remission cohort
├── 4_sklearn.ipynb                # Traditional ML models implementation
├── 5_sklearn_biochemical_remission.ipynb  # ML models for remission cohort
├── 6_traditional_stats_benchmarking.ipynb  # Comparison with statistical methods
├── 7_validation_spain.ipynb       # External validation on Spanish cohort
├── 8_validation_australia.ipynb   # External validation on Australian cohort
├── 9_validation_scotland.ipynb    # External validation on Scottish cohort
├── 10_validation_combined.ipynb   # Combined validation analysis
├── pipeline_summary.png           # Visual overview of the ML pipeline
├── requirements.txt               # Package dependencies
└── LICENSE                        # Apache License 2.0
```

## 📓 Notebook Descriptions

| Notebook | Description |
|----------|-------------|
| 1_demographics.ipynb | Analysis of cohort characteristics and statistical comparisons |
| 2_dnn.ipynb | Implementation of deep neural network for fatigue prediction using TensorFlow |
| 3_dnn_biochemical_remission.ipynb | DNN model specifically for patients in biochemical remission |
| 4_sklearn.ipynb | Implementation of SKLearn ML models for the full IBD cohort |
| 5_sklearn_biochemical_remission.ipynb | SKLearn ML models for patients in biochemical remission |
| 6_traditional_stats_benchmarking.ipynb | Traditional statistical analysis for comparison with ML approaches |
| 7_validation_spain.ipynb | External validation using Spanish cohort data |
| 8_validation_australia.ipynb | External validation using Australian cohort data |
| 9_validation_scotland.ipynb | External validation using Scottish cohort data |
| 10_validation_combined.ipynb | Combined analysis of all validation cohorts |

## 🔍 Validation Datasets

The models are validated on three independent external cohorts:

1. **Scottish Cohort**: Data collected from IBD patients in Edinburgh not used in model building
2. **Spanish Cohort**: Independent validation set with similar clinical metrics but different units
3. **Australian Cohort**: Third validation cohort for additional geographical diversity

## 🚀 Usage

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

## 📊 Output Directory

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
│   ├── validation                    # External validation results
│   │   ├── spain                     # Spanish cohort validation
│   │   ├── australia                 # Australian cohort validation
│   │   ├── scotland                  # Scottish cohort validation
│   │   └── combined                  # Combined validation analysis
│   └── biochemical_remission         # Results for remission cohort
│       └── ...                       # Similar structure as all_ibd
```

## 🧠 Model Interpretability

SHAP (SHapley Additive exPlanations) values are used to interpret the models:

- Summary plots show the most influential features
- Force plots visualize individual predictions
- Comparison of feature importance between different models and cohorts

## 🔄 Reproducing Our Results

To reproduce our results:

1. Run the notebooks in numerical order (1_demographics.ipynb → 2_dnn.ipynb → etc.)
2. Each notebook saves outputs to the appropriate directory in output
3. Validation notebooks (7-10) rely on trained models from earlier notebooks

## 📝 Citation

If you use code or models from this repository in your research, please cite:

```plaintext
[citation details to be added after publication]
```

## 💬 Feedback and Contributions

For questions, feedback, or contributions, please:

- Open an issue on GitHub
- Submit a pull request with proposed changes
- Contact the authors directly

## 📧 Contact

- Shaun Chuah ([shaun.chuah@glasgow.ac.uk](mailto:shaun.chuah@glasgow.ac.uk))
- [Gut Translational Research Group](https://www.gla.ac.uk/schools/infectionimmunity/staff/gwotzerho/theguttranslationalresearchgroup/)

## 👨‍💻 Repository Authors

- Shaun Chuah ([github.com/shaunchuah](https://github.com/shaunchuah))
- Robert Whelan ([github.com/rw509](https://github.com/rw509))

## ⚖️ License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
