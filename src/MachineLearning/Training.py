# -*- coding utf-8, LF -*-

import os
import pickle
import sys
from datetime import datetime, timedelta
from logging import getLogger

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

sys.path.append(os.getcwd() + "/src")

from config.params import ML_MODEL_DIR, ML_RESULT_DIR
from MachineLearning.basemodel import BaseModel


class SupervisedML(BaseModel):
    logger = getLogger("main").getChild("Machine Learning")

    def __init__(self, parameter: str) -> None:
        super().__init__(parameter)
        self.param_dict["clf_params"] = dict()
        self.param_dict["model_name"] = None

    @classmethod
    def load_model(cls, parameter: str, mode: str = "mixsep", name: str = "LinearSVC", C: float = 1, randomstate: int = 100, path=None):  # noqa: ANN206
        if path is None:
            path = ML_MODEL_DIR + f"/model/model_{name}_{parameter}_{mode}.C={C}.randomstate={randomstate}.sav"

        model = cls(parameter)
        model.model = pickle.load(open(path, "rb"))
        model.param_dict["mode"] = mode
        model.param_dict["parameter"] = parameter
        model.param_dict["model_name"] = name
        model._load_npys(mode=mode, parameter=parameter)
        return model

    def kneighbors(self) -> KNeighborsClassifier:
        self.param_dict["model_name"] = "KNeighbors"

        n_neighbors = int(np.sqrt(self.X_train.shape[0]))  # kの設定
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def linearSVC(self, C: float = 1, randomstate: int = 0) -> LinearSVC:
        self.param_dict["model_name"] = "LinearSVC"
        self.param_dict["clf_params"]["C"] = C
        self.param_dict["clf_params"]["randomstate"] = randomstate

        self.model = LinearSVC(C=C, random_state=randomstate)  # インスタンスを生成
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def rbfSVC(self, C: float = 1, gamma: float = 1, randomstate: int = 100, cls="ovo") -> SVC | OneVsRestClassifier:
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

    def XGBoost(self, n_estimators=80, max_depth=4, gamma=3, params: dict | None=None) -> XGBClassifier:
        self.param_dict["model_name"] = "XGBoost"
        self.param_dict["clf_params"]["gamma"] = gamma
        self.param_dict["clf_params"]["max_depth"] = max_depth
        self.param_dict["clf_params"]["n_estimators"] = n_estimators

        tree_methods = "gpu_hist"
        # if GPU:
        #     tree_methods = "gpu_hist"
        # else:
        #     tree_methods = "hist"
        self.model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, gamma=gamma, tree_method=tree_methods)

        if params is not None:
            self.model.set_params(**params)

        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def predict(self) -> None:
        f = open(
            ML_RESULT_DIR + f"/{self.param_dict['model_name']}/{self.param_dict['parameter']}_{self.param_dict['mode']}.txt",
            "a",
            encoding="utf-8",
        )
        self.pred = self.model.predict(self.X_test)
        time = datetime.strftime(datetime.now() + timedelta(hours=9), "%Y-%m-%d %H:%M:%S")
        print(time, file=f)
        print("パラメータ : ", self._dict_to_str("clf_params"), "\n", file=f)
        print("スコア     : ", self.model.score(self.X_test, self.y_test), file=f)
        print("正解率     : ", accuracy_score(self.y_test, self.pred), file=f)
        print("適合率     : ", precision_score(self.y_test, self.pred, average="macro"), file=f)
        print("再現率     : ", recall_score(self.y_test, self.pred, average="macro"), file=f)
        print("F1値       : ", f1_score(self.y_test, self.pred, average="macro"), file=f)
        print("混合行列   : \n", confusion_matrix(self.y_test, self.pred), file=f)
        print("要約       : \n", classification_report(self.y_test, self.pred), "\n\n\n", file=f)

        f.close()

    def get_params(self) -> dict:
        return self.param_dict

    def save_model(self, model_path: str | None = None) -> None:
        if model_path is None:
            model_path = (
                ML_MODEL_DIR
                + f"/model/model_{self.param_dict['model_name']}_{self.param_dict['parameter']}_{self.param_dict['mode']}{self._dict_to_str()}.sav"
            )

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            f.close()
