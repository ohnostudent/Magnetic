# -*- coding utf-8, LF -*-

import os
import pickle
import sys
from logging import getLogger

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

sys.path.append(os.getcwd() + "/src")

from config.params import ML_MODEL_DIR
from MachineLearning.basemodel import BaseModel


class SupervisedML(BaseModel):
    logger = getLogger("main").getChild("Machine Learning")

    def __init__(self, parameter) -> None:
        super().__init__(parameter)
        self.param_dict["clf_params"] = dict()
        self.param_dict["model_name"] = None

    def kneighbors(self):
        self.param_dict["model_name"] = "KNeighbors"

        n_neighbors = int(np.sqrt(self.X_train.shape[0]))  # kの設定
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def linearSVC(self, C, randomstate):
        self.param_dict["model_name"] = "LinearSVC"
        self.param_dict["clf_params"]["C"] = C
        self.param_dict["clf_params"]["randomstate"] = randomstate

        self.model = LinearSVC(C=C, random_state=randomstate)  # インスタンスを生成
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def rbfSVC(self, C, gamma, randomstate, cls="ovo"):
        self.param_dict["model_name"] = f"rbfSVC_{cls}"
        self.param_dict["clf_params"]["C"] = C
        self.param_dict["clf_params"]["gamma"] = gamma
        self.param_dict["clf_params"]["randomstate"] = randomstate

        match cls:
            case "ovo":
                self.model = SVC(C=C, kernel="rbf", random_state=randomstate)  # インスタンスを生成
            case "ovr":
                estimator = SVC(C=C, kernel="rbf", gamma=gamma)
                self.model = OneVsRestClassifier(estimator)
            case _:
                raise ValueError

        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def get_params(self):
        return self.param_dict

    def predict(self):
        self.pred = self.model.predict(self.X_test)
        self.model.score(self.X_test, self.y_test)
        print("正解率   : ", accuracy_score(self.y_test, self.pred))
        print("適合率   : ", precision_score(self.y_test, self.pred, average="macro"))
        print("再現率   : ", recall_score(self.y_test, self.pred, average="macro"))
        print("F1値     : ", f1_score(self.y_test, self.pred, average="macro"))
        print("混合行列 : \n", confusion_matrix(self.y_test, self.pred))
        print("要約     : \n", classification_report(self.y_test, self.pred))
        print()

    @classmethod
    def load_model(
        cls, parameter: str, mode: str = "mixsep", name: str = "LinearSVC", C: float = 0.3, randomstate: int = 0, path=None
    ):  # noqa: ANN206
        if path is None:
            path = ML_MODEL_DIR + f"/model/model_{name}_{parameter}_{mode}.C={C}.randomstate={randomstate}.sav"

        model = cls(parameter)
        model.model = pickle.load(open(path, "rb"))
        model.param_dict["mode"] = mode
        model.param_dict["parameter"] = parameter
        model.param_dict["model_name"] = name
        model._load_npys(mode=mode, parameter=parameter)
        return model

    def save_model(self, model_path=None) -> None:
        if model_path is None:
            model_path = (
                ML_MODEL_DIR
                + f"/model/model_{self.param_dict['model_name']}_{self.param_dict['parameter']}_{self.param_dict['mode']}{self._dict_to_str()}.sav"
            )

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            f.close()

    def grid_search(self, clf, param_grid, cv, scoring="roc_auc", n_jobs=-1, verbose=3):
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
        grid_search.fit(self.X_train, self.y_train)

        print("Best parameter:", grid_search.best_score_)
        print("Score (train):", grid_search.score(self.X_train, self.y_train))
        print("Score (test):", grid_search.score(self.X_test, self.y_test))

        return grid_search
