# -*- coding utf-8, LF -*-

import os
import sys
from logging import getLogger

import optuna
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from torch import cuda
from xgboost import XGBClassifier

sys.path.append(os.getcwd() + "/src")

from config.params import ML_MODEL_DIR
from config.SetLogger import logger_conf
from MachineLearning.basemodel import BaseModel


class TuningOptuna:
    def __init__(self) -> None:
        self.logger = getLogger("Tuning_optuna").getChild("Tuning")
        self.best_parameter_dict: dict[str, dict] = {}

    def load(self, mode: str = "mix", parameter: str = "enstrophy", clf_name: str = "", label: int = 0, randomstate: int = 42):
        self.mode = mode
        if mode == "sep":
            self.label = label
            self.mode_name = mode + str(label)
        else:
            self.mode_name = mode
        self.parameter = parameter
        self.clf_name = clf_name
        self._clf_methods()

        if mode == "sep":
            self.label = label
        else:
            self.label = None

        self.logger.debug("SAVE", extra={"addinfo": f"mode={mode}, parameter={parameter}, label={label}"})
        self.model = BaseModel.load_npys(mode=mode, parameter=parameter, label=label, random_state=randomstate)

        return self.model

    def _set_params(self, trial):
        match self.clf_name:
            case "kNeighbors":
                tuning_params = {
                    "n_neighbors": trial.suggest_int("n_neighbors", 3, 500),
                    "p": trial.suggest_float("p", 1, 2),
                }

            case "LinearSVC":
                tuning_params = {
                    "C": trial.suggest_float("C", 0.001, 500),
                }

            case "rbfSVC":
                tuning_params = {"C": trial.suggest_float("C", 0.01, 500), "gamma": trial.suggest_float("gamma", 0.01, 10), "kernel": "rbf", "tol": 0.001, "random_state": 42, "verbose": 0}

            case "XGBoost":
                tuning_params = {
                    "max_depth": trial.suggest_int("max_depth", 1, 10),
                    "eta": trial.suggest_float("eta", 0.001, 0.3),
                    "gamma": trial.suggest_float("gamma", 0.01, 10),
                    "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10),
                    "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 2.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2.0),
                    "tree_method": "gpu_hist" if cuda.is_available() else "hist",
                    "n_estimators": 500,
                    "n_jobs": -1,
                    "random_state": 42,
                    "eval_metric": "auc",
                }
            case _:
                raise ValueError

        return tuning_params

    def _clf_methods(self):
        match self.clf_name:
            case "kNeighbors":
                self.clf = KNeighborsClassifier
            case "LinearSVC":
                self.clf = LinearSVC
            case "rbfSVC":
                self.clf = lambda **x: OneVsRestClassifier(SVC(**x))
            case "XGBoost":
                self.clf = XGBClassifier
                self.eval_set = [(self.model.X_train, self.model.y_train)]
            case _:
                raise ValueError

        return self.clf

    def objective(self, trial):
        tuning_params = self._set_params(trial)
        clf = self.clf(**tuning_params)

        clf.fit(self.model.X_train, self.model.y_train)
        pred_test = clf.predict_proba(self.model.X_test)  # type: ignore
        return roc_auc_score(self.model.y_test, pred_test, multi_class="ovr")

    def load_study(self, study_name: str | None = None):
        if study_name is None:
            study_name = f"optuna_{self.clf_name}_{self.parameter}_{self.mode_name}"
        storage = "sqlite:///" + ML_MODEL_DIR + f"/tuning/{self.mode_name}/optuna_{self.clf_name}_{self.parameter}_{self.mode_name}.sav"

        self.study = optuna.load_study(study_name=study_name, storage=storage)
        return self.study

    def create_study(self, study_name: str | None = None):
        if study_name is None:
            study_name = f"optuna_{self.clf_name}_{self.parameter}_{self.mode_name}"
        storage = "sqlite:///" + ML_MODEL_DIR + f"/tuning/{self.mode_name}/optuna_{self.clf_name}_{self.parameter}_{self.mode_name}.sav"
        sampler = optuna.samplers.TPESampler(seed=42)

        self.study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name, storage=storage, load_if_exists=True)

    def do_optimizer(self):
        self.study.optimize(lambda x: self.objective(x), n_trials=30)  # type: ignore

    def plot(self):
        optuna.visualization.plot_optimization_history(self.study)

    def save(self):
        # 最適パラメータの表示と保持
        best_params = self.study.best_trial.params
        best_score = self.study.best_trial.value
        self.best_parameter_dict[self.parameter] = best_params
        self.logger.debug("PARAMETER", extra={"addinfo": f"{best_params}"})


if __name__ == "__main__":
    # ベイズ最適化を実行
    logger = logger_conf("Tuning_optuna")
    logger.debug("START", extra={"addinfo": "処理開始"})

    from logging import FileHandler

    from config.params import LOG_DIR

    optuna.logging.get_logger("optuna").addHandler(FileHandler(LOG_DIR + "/optuna.log"))

    tuning_best_param_dict = dict()
    clf_name = "rbfSVC"  # kNeighbors, LinearSVC, rbfSVC, XGBoost
    # mode = "mix"  # sep, mixsep, mix
    # parameter = "density"  # density, energy, enstrophy, pressure, magfieldx, magfieldy, velocityx, velocityy
    # label = 0

    tu = TuningOptuna()
    for mode in ["mixsep", "mix", "sep"]:
        if mode == "sep":
            for label in range(3):
                mode_name = mode + str(label)
                for parameter in ["density", "energy", "enstrophy", "pressure", "magfieldx", "magfieldy", "velocityx", "velocityy"]:
                    try:
                        tu.load(clf_name=clf_name, mode=mode, parameter=parameter, label=label)
                        tu.create_study()
                        tu.do_optimizer()
                        tu.save()
                    except Exception as e:
                        logger.error("ERROR", extra={"addinfo": f"{e}"})
                tuning_best_param_dict[mode_name] = tu.best_parameter_dict

        else:
            for parameter in ["enstrophy", "pressure", "magfieldx", "magfieldy", "velocityx", "velocityy"]:
                # for parameter in ["density", "energy", "enstrophy", "pressure", "magfieldx", "magfieldy", "velocityx", "velocityy"]:
                try:
                    tu.load(clf_name=clf_name, mode=mode, parameter=parameter)
                    tu.create_study()
                    tu.do_optimizer()
                    tu.save()
                except Exception as e:
                    logger.error("ERROR", extra={"addinfo": f"{e}"})
            tuning_best_param_dict[mode] = tu.best_parameter_dict

    import json

    with open("./params.json", "w", encoding="utf-8") as f:
        json.dump(tuning_best_param_dict, f)

    logger.debug("END", extra={"addinfo": "処理終了"})
