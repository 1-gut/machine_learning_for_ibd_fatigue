import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config.constants import RANDOM_SEED

random_seed = RANDOM_SEED

models_and_params = {
    "Random Forest": (
        RandomForestClassifier(random_state=random_seed),
        {
            "n_estimators": [1000, 2000],
            "max_depth": [None, 5, 20],
            "min_samples_split": [2, 10, 20],
        },
    ),
    "Logistic Regression": (
        LogisticRegression(random_state=random_seed, max_iter=10000),
        {"C": [0.01, 1, 10], "penalty": ["l2"], "solver": ["lbfgs", "saga"]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=random_seed, algorithm="SAMME"),
        {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1]},
    ),
    "XGBoost": (
        xgb.XGBClassifier(random_state=random_seed, objective="binary:logistic"),
        {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
        },
    ),
    "SVC": (
        SVC(probability=True, random_state=random_seed),
        {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 1]},
    ),
    "MLPClassifier": (
        MLPClassifier(max_iter=10000, random_state=random_seed),
        {
            "hidden_layer_sizes": [(100, 50), (100,), (90,), (110,)],
            "activation": ["tanh", "relu"],
            "solver": ["sgd", "adam"],
        },
    ),
}
