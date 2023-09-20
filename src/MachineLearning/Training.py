# -*- coding utf-8, LF -*-

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

sys.path.append(os.getcwd() + "/src")

from MachineLearning.basemodel import BaseModel


class SupervisedML(BaseModel):
    def __init__(self, parameter) -> None:
        super().__init__(parameter)

    def kneighbors(self):
        n_neighbors = int(np.sqrt(6000))  # kの設定
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model_returns()

    def rbfSVC(self, randomstate):
        self.model = SVC(C=0.3, kernel="rbf", random_state=randomstate)  # インスタンスを生成
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model_returns()

    def linearSVC(self, randomstate):
        self.model = LinearSVC(C=0.3, random_state=randomstate)  # インスタンスを生成
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model_returns()

    def model_returns(self):
        pred = self.model.predict(self.X_test)
        ml_res = pd.DataFrame(np.array([self.test_paths, self.y_test, pred]).T, columns=["path", "y", "predict"])
        report = classification_report(self.y_test, pred)
        return self.model, ml_res, report
