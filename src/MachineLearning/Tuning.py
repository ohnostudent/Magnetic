# -*- coding utf-8, LF -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve, validation_curve
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

sys.path.append(os.getcwd() + "/src")

from config.params import ML_PARAM_DICT, dict_to_str
from MachineLearning.Training import SupervisedML

cv = KFold(n_splits=5, shuffle=True)
param_grid = {"linearSVC": {"C": [0.001, 0.01, 0.1, 1, 10, 100]}, "rbfSVC": {"gamma": [0.01, 0.1, 1, 10], "C": [0.001, 0.01, 0.1, 1, 10, 100]}}
grid_search_params = {"scoring": "roc_auc", "cv": 5, "verbose": 3, "n_jobs": -1}
cross_val_params = {"cv": cv, "scoring": "accuracy", "n_jobs": -1}


def _set_classifier(clf_name):
    match clf_name:
        case "KMeans":
            clf = KMeans
        case "KNeighbors":
            clf = KNeighborsClassifier
        case "LinearSVC":
            clf = LinearSVC
        case "rgfSVC":
            clf = SVC
        case "XGboost":
            clf = XGBClassifier
        case _:
            raise ValueError
    return clf


def plot_learning_curve(clf_name: str, clf_params: dict, model: SupervisedML):
    clf = _set_classifier(clf_name)
    estimator = clf(**clf_params)  # 分類器
    train_sizes, train_scores, test_scores, _, _ = learning_curve(estimator=estimator, X=model.X_train, y=model.y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3, n_jobs=5)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # グラフに可視化
    plt.figure(figsize=(12, 8))

    plt.plot(train_sizes, train_mean, marker="o", label="Train accuracy")
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.2)

    plt.plot(train_sizes, test_mean, marker="s", linestyle="--", label="Validation accuracy")
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.2)

    plt.grid()
    plt.title("Learning curve", fontsize=16)
    plt.xlabel("Number of training data sizes", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(f"./ML/result/plot/learning_curve.{clf_name}.{dict_to_str(clf_params)}.jpg")
    plt.show()


def grid_search_cv(model, clf_name: str, clf_params: dict, param_grid: dict = param_grid, grid_search_params: dict = grid_search_params, cross_val: bool = False):
    clf = _set_classifier(clf_name)
    clf = clf(**clf_params)
    estimator = GridSearchCV(clf, param_grid[clf_name], **grid_search_params)

    if cross_val:
        return estimator

    estimator.fit(model.X_train)
    return estimator


def cross_validation(model, clf_name: str, clf_params: dict, param_grid: dict, grid_search_params: dict, cross_val_params: dict):
    estimator = grid_search_cv(model, clf_name, clf_params, param_grid, grid_search_params)
    cv_scores = cross_val_score(estimator, model.X_train, model.y_train, **cross_val_params)
    print("cv_scores:", cv_scores, "\nmean:", cv_scores.mean())


if __name__ == "__main__":
    model = SupervisedML.load_npys(mode="all", parameter="density")
    clf_name = "rbfSVC"
    clf_params = ML_PARAM_DICT[clf_name]

    cross_validation(model, clf_name, clf_params, param_grid, grid_search_params, cross_val_params)
