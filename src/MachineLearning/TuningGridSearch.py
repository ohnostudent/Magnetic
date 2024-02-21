# -*- coding utf-8, LF -*-

"""
GridSearch

線形SVM, k近傍法のパラメータチューニング

"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

sys.path.append(os.getcwd() + "/src")

from config.params import dict_to_str
from MachineLearning.Training import SupervisedML


class TuningGrid:
    def __init__(self, training_parameter, split_mode) -> None:
        self.model = SupervisedML.load_npys(training_parameter=training_parameter, split_mode=split_mode)
        self.cv = KFold(n_splits=5, shuffle=True)

    def _set_classifier(self) -> type[KNeighborsClassifier] | type[LinearSVC] | type[SVC]:
        match self.clf_name:
            case "kNeighbors":
                clf = KNeighborsClassifier
            case "LinearSVC":
                clf = LinearSVC
            # case "rbfSVC":
            #     clf = SVC
            case _:
                raise ValueError
        return clf

    def set_params(self, clf_name: str, param_grid: dict | None = None, grid_search: dict | None = None, cross_val: dict | None = None):
        self.clf_name = clf_name
        self.clf = self._set_classifier()

        if param_grid is None:
            self.param_grid = {"kNeighbors": {"n_neighbors": [3, 10, 50, 100, 500, 1000]},"LinearSVC": {"C": [0.001, 0.01, 0.1, 1, 10, 100]}, "rbfSVC": {"gamma": [0.01, 0.1, 1, 10, 100], "C": [0.01, 0.1, 1, 10, 100]}}
        else:
            self.param_grid = param_grid

        if grid_search is None:
            self.grid_search_params = {"scoring": "roc_auc", "cv": 5, "verbose": 3, "n_jobs": 3}
        else:
            self.grid_search_params = grid_search

        if cross_val is None:
            self.cross_val_params = {"cv": self.cv, "scoring": "auc", "n_jobs": -1}
        else:
            self.cross_val_params = cross_val

    def _set_estimator(self, clf_params, GS: bool = True) -> GridSearchCV[KNeighborsClassifier | LinearSVC | SVC] | KNeighborsClassifier | LinearSVC | SVC:
        self.clf = self.clf(**clf_params) # type: ignore

        if GS:
            estimator = GridSearchCV(self.clf, self.param_grid[self.clf_name], **self.grid_search_params)
        else:
            estimator = self.clf
        return estimator


    def grid_search_cv(self, clf_params: dict):
        """
        グリッドサーチを行う関数

        Args:
            self.clf_name (str): 分類器名
            clf_params (dict): 学習用パラメータ
            param_grid (dict, optional): 検証したいパラメータ. Defaults to param_grid.
            grid_search_params (dict, optional): グリッドサーチ用パラメータ. Defaults to grid_search_params.
            cross_val (bool, optional): クロスバリデーション用. Defaults to False.

        Returns:
            GridSearchCV[KNeighborsClassifier | LinearSVC | SVC]
        """
        estimator = self._set_estimator(clf_params, GS=True)
        estimator.fit(self.model.X_train, self.model.y_train)
        return estimator

    def cross_validation(self, clf_params: dict) -> None:
        """交差検証

        Args:
            self.clf_name (str): 分類器名
            clf_params (dict): 学習用パラメータ
            param_grid (dict, optional): 検証したいパラメータ.
            grid_search_params (dict, optional): グリッドサーチ用パラメータ.
            cross_val_params (dict): 交差検証に用いるパラメータ
            use_grid (bool, optional): GridSearch を用いるかどうか. Defaults to True.
        """
        estimator = self._set_estimator(clf_params)
        cv_scores = cross_val_score(estimator, self.model.X_train, self.model.y_train, **self.cross_val_params)
        print("cv_scores:", cv_scores)
        print("mean:", cv_scores.mean())

    def calc_learning_curve(self, clf_params: dict):
        """
        学習曲線を描く

        Args:
            self.clf_name (str): 分類器名.
            clf_params (dict): 学習用パラメータ.
        """
        estimator = self._set_estimator(clf_params)
        train_sizes, train_scores, test_scores, _, _ = learning_curve(estimator=estimator, X=self.model.X_train, y=self.model.y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3, n_jobs=5)
        self._plot_learning_curve(train_sizes, train_scores, test_scores)

    def _plot_learning_curve(self, train_sizes, train_scores, test_scores):
        # グラフに可視化
        plt.figure(figsize=(12, 8))

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes, train_mean, marker="o", label="Train accuracy")
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.2)

        plt.plot(train_sizes, test_mean, marker="s", linestyle="--", label="Validation accuracy")
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.2)

        plt.grid()
        plt.title("Learning curve", fontsize=16)
        plt.xlabel("Number of training data sizes", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig(f"./ML/result/plot/learning_curve.{self.clf_name}.{dict_to_str(clf_params)}.jpg")
        plt.show()


if __name__ == "__main__":
    training_parameter = "density"
    split_mode = "all"
    clf_name = "linearSVC"  # kneighbors, linearSVC, rbfSVC, XGBoost

    tg = TuningGrid(training_parameter, split_mode)
    # tg.grid_search_cv(clf_name, clf_params, param_grid, grid_search_params)
    # tg.cross_validation(clf_name, clf_params, param_grid, grid_search_params, cross_val_params, use_grid=True)
