# -*- coding utf-8, LF -*-

"""
学習用

"""

import os
import pickle
import sys
from datetime import datetime
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from torch import cuda
from xgboost import XGBClassifier

sys.path.append(os.getcwd() + "/src")

from config.params import ML_MODEL_DIR, ML_RESULT_DIR, dict_to_str
from MachineLearning.basemodel import BaseModel


class SupervisedML(BaseModel):
    CUDA = cuda.is_available()
    if CUDA:
        cuda.empty_cache()

    def __init__(self, training_parameter: str) -> None:
        super().__init__(training_parameter)
        self.logger = getLogger("ML").getChild(training_parameter)
        self.set_default(training_parameter)

    def __str__(self) -> str:
        params = dict_to_str(self.param_dict["train_params"], sep="\n\t")
        doc = f"""
        Params:
            {params}
        """
        return doc

    @classmethod
    def load_model(cls, training_parameter: str, split_mode: str, split_mode_label: int | None = None, clf_name: str = "LinearSVC", model_random_state: int = 42, load_path: str | None = None):  # noqa: ANN206
        model = cls(training_parameter)
        model.set_default(training_parameter)
        model.split_mode_name = split_mode + str(split_mode_label) if split_mode == "sep" else split_mode
        model.param_dict["split_mode"] = split_mode
        model.param_dict["training_parameter"] = training_parameter
        model.param_dict["clf_name"] = clf_name

        if split_mode_label is not None:
            model.param_dict["split_mode_label"] = split_mode_label
        if load_path is None:
            load_path = ML_MODEL_DIR + f"/model/{split_mode}/model_{clf_name}_{training_parameter}_{split_mode}.sav"

        model.PROBA = clf_name in ["kNeighbors", "rbfSVC", "XGBoost"]
        model._load_npys(split_mode=split_mode, training_parameter=training_parameter, random_state=model_random_state, split_mode_label=split_mode_label)
        model.model = pickle.load(open(load_path, "rb"))
        return model

    def do_learning(self, clf_name: str, tuning_params, fixed_params: dict):
        """学習を行う関数

        Args:
            clf_name (str): 使用する分類器の名前 (kNeighbors / LinearSVC / rbfSVC / XGBoost)
            param_dict(dict): 各種パラメータを格納する辞書, 各値は以下の通り.
            - kNeighbors
                - n_clusters            : クラスターの個数
                - init                  : セントロイドの初期値をランダムに設定  default: 'k-means++'
                - n_init                : 異なるセントロイドの初期値を用いたk-meansの実行回数 default: '10' 実行したうちもっとSSE値が小さいモデルを最終モデルとして選択
                - max_iter              : k-meansアルゴリズムの内部の最大イテレーション回数  default: '300'
                - tol                   : 収束と判定するための相対的な許容誤差 default: '1e-04'
                - random_state          : セントロイドの初期化に用いる乱数発生器の状態

            - LinearSVC
                - C (float)             : マージン. Defaults to 1.
                - random_state (int)     : 乱数のseed値. Defaults to 100.

            - rbfSVC
                - C (float)             : マージン. Defaults to 1.
                - gamma (float)         : ガンマ値. Defaults to 3.
                - cls (str)             : 非線形SVM の種類の指定(ovo / ovr). Defaults to "ovo".
                - random_state (int)     : 乱数のseed値. Defaults to 100.

            - XGBoost
                - n_estimators (int)    : 学習する決定木の数. Defaults to 1000.
                - max_depth (int)       : 決定木の深さ. Defaults to 4.
                - learning_rate		    : shrinkage
                - subsample		    	:
                - colsample_bytree	    :
                - missing				:
                - eval_metric			: ブースティング時の各イテレーション時に使う評価指標
                - tree_method			: 木構造の種類 (hist, gpu_hist)
                - random_state          : 乱数の seed値
                - params (dict | None)  : その他使用するパラメータ. Defaults to None.
        """
        self.logger.debug("START", extra={"addinfo": "学習開始"})
        self.param_dict["clf_name"] = clf_name
        self.param_dict["tuning_params"] = tuning_params
        self.param_dict["fixed_params"] = fixed_params

        match clf_name:
            case "kNeighbors":
                self.PROBA = True
                self.model = self.kNeighbors(tuning_params, fixed_params)
            case "LinearSVC":
                self.PROBA = False
                self.model = self.LinearSVC(tuning_params, fixed_params)
            case "rbfSVC":
                self.PROBA = True
                self.model = self.rbfSVC(tuning_params, fixed_params)
            case "XGBoost":
                self.PROBA = True
                self.model = self.XGBoost(tuning_params, fixed_params)
            case _:
                raise ValueError

        self.logger.debug("END", extra={"addinfo": "学習終了"})
        return self.model

    def kNeighbors(self, tuning_params: dict, fixed_params: dict) -> KNeighborsClassifier:
        """_summary_

        Args:
            "algorithm": "auto",  // 最近傍値を計算するアルゴリズム(auto, ball_tree, kd_tree, brute)
            "leaf_size": 30,  // BallTree または KDTree に渡される葉のサイズ
            "metric": "minkowski",  // 距離の計算に使用するメトリック
            "metric_params": null,  // メトリック関数の追加のパラメータ
            "n_jobs": null,  // 実行する並列ジョブの数
            "n_neighbors": 5,  // k の数
            "p": 2,  // Minkowski メトリクスの検出力パラメータ
            "weights": "uniform"  // 重み
        Returns:
            KNeighborsClassifier: _description_
        """
        self.model = KNeighborsClassifier(**tuning_params, **fixed_params)
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def LinearSVC(self, tuning_params: dict, fixed_params: dict) -> LinearSVC:
        """_summary_

        Args:
            "C": 0.39,  // 正則化パラメータ、マージン
            "dual": "auto",  // 双対最適化問題または主最適化問題を解決するアルゴリズム
            "fit_intercept": true,  // 切片を適合させるかどうか
            "intercept_scaling": 1,
            "loss": "squared_hinge",  // 損失関数(hinge, squared_hinge)
            "max_iter": 1000,  // 実行される反復の最大数
            "multi_class": "ovr",  // マルチクラス戦略
            "penalty": "l2",  // ペナルティ
            "random_state": null,  // 乱数の seed値
            "tol": 0.0001,  // 停止基準の許容値
            "verbose": 5  // 詳細な出力を有効
        Returns:
            LinearSVC
        """
        self.model = LinearSVC(**tuning_params, **fixed_params)  # インスタンスを生成
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def rbfSVC(self, tuning_params: dict, fixed_params: dict) -> SVC:
        """_summary_

        Args:
            "C": 1.0,  // 正則化パラメータ、マージン
            "cache_size": 200,  // キャッシュサイズ
            "coef0": 0.0,  // Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
            "decision_function_shape": "ovr",
            "degree": 3,  // 多項式(poly)カーネルの次数
            "gamma": "scale",  // カーネルの係数、ガウスカーネル(rbf): 1/(n_features * X.var()) と シグモイドカーネル(sigmoid): 1 /n_features
            "kernel": "rbf",  // カーネル('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            "max_iter": -1,  // ソルバー内の反復に対するハード制限
            "probability": false,  // true の場合、予測時に各クラスに属する確率を返す
            "random_state": null,  // 乱数の seed値
            "shrinking": true,  // 縮小ヒューリスティックを使用するかどうか
            "tol": 0.001,  // 停止基準の許容値
            "verbose": 5  // 詳細な出力を有効

        Returns:
            SVC
        """
        self.model = SVC(**tuning_params, **fixed_params)
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def XGBoost(self, tuning_params: dict, fixed_params: dict, early_stopping_rounds: int = 100) -> XGBClassifier:
        """parameters

        Args:
            eval_metric: "auc",
            n_estimators: 500,
            n_jobs: -1, // 並列数
            random_state: 42, // 乱数
            early_stopping_rounds: int = 100
        Returns:
            XGBClassifier
        """
        if self.CUDA:
            tree_method = "gpu_hist"
        else:
            tree_method = "hist"

        self.param_dict["tuning_params"]["tree_method"] = tree_method
        self.model = XGBClassifier(**tuning_params, **fixed_params)

        # if params is not None:
        self.model.set_params(tree_method=tree_method, early_stopping_rounds=early_stopping_rounds)

        eval_set = [(self.X_train, self.y_train)]
        self.model.fit(model.X_train, model.y_train, eval_set=eval_set, verbose=50)  # モデルの学習

        return self.model

    def predict(self, pred_path: str | np.ndarray | None = None) -> None:
        """
        予測を行う関数

        Args:
            pred_path (str | np.ndarray | None, optional): 予測を行うデータ. Defaults to None.

        Raises:
            ValueError: _description_
        """
        self.logger.debug("PREDICT", extra={"addinfo": "予測\n"})

        # 事前に分割したテストデータを使用する場合
        if pred_path is None:
            self.pred = self.model.predict(self.X_test)
            if self.PROBA:
                self.pred_proba = self.model.predict_proba(self.X_test)  # type: ignore

        # パスを指定する場合
        elif isinstance(pred_path, str):
            self.pred = self.model.predict(np.load(pred_path))
            if self.PROBA:
                self.pred_proba = self.model.predict_proba(np.load(pred_path))  # type: ignore

        # 読み込んだもので予測する場合
        elif isinstance(pred_path, np.ndarray):
            self.pred = self.model.predict(pred_path)
            if self.PROBA:
                self.pred_proba = self.model.predict_proba(pred_path)  # type: ignore

        else:
            raise ValueError(f"{type(pred_path)} is not supported.")

    def print_scores(self):
        """
        評価データの出力
        """
        save_path = ML_RESULT_DIR + f"/{self.param_dict['clf_name']}/{self.param_dict['split_mode']}_{self.param_dict['training_parameter']}.txt"

        f = open(save_path, "a", encoding="utf-8")
        time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        print(f"【 {time} 】", file=f)
        print("変数       :", self.param_dict["training_parameter"], self.param_dict["split_mode"], file=f)
        print("パラメータ :\n", dict_to_str(self.model.get_params(), sep="\n"), "\n", file=f)
        print("trainスコア:", self.model.score(self.X_train, self.y_train), file=f)
        print("test スコア:", self.model.score(self.X_test, self.y_test), file=f)

        acc_score = accuracy_score(self.y_test, self.pred)
        print("正解率     :", acc_score, file=f)
        if self.PROBA:
            auc_score = roc_auc_score(self.y_test, self.pred_proba, multi_class="ovr")
            print("AUC        :", auc_score, file=f)

        print("適合率macro:", precision_score(self.y_test, self.pred, average="macro"), file=f)
        print("再現率macro:", recall_score(self.y_test, self.pred, average="macro"), file=f)
        f1_macro = f1_score(self.y_test, self.pred, average="macro")
        print("F1値  macro:", f1_macro, file=f)

        print("適合率micro:", precision_score(self.y_test, self.pred, average="micro"), file=f)
        print("再現率micro:", recall_score(self.y_test, self.pred, average="micro"), file=f)
        f1_micro = f1_score(self.y_test, self.pred, average="micro")
        print("F1値  micro:", f1_micro, file=f)

        clf_rep = confusion_matrix(self.y_test, self.pred)
        self._plot_confusion_matrix(clf_rep)
        print("混合行列   : \n", clf_rep, file=f)

        print("要約       : \n", classification_report(self.y_test, self.pred, digits=3), file=f)
        print("\n\n\n", file=f)
        f.close()
        self.logger.debug("SCORE", extra={"addinfo": f"acc_score={acc_score}, f1_macro={f1_macro}"})

    def _plot_confusion_matrix(self, clf_rep):
        """
        混同行列の可視化
        """
        fig, ax = plt.subplots(1, 1)
        cmp = ConfusionMatrixDisplay(clf_rep)  # type: ignore
        ax.set_title(f"{self.param_dict['clf_name']}_{self.param_dict['training_parameter']}_{self.split_mode_name}")
        cmp.plot(cmap=plt.cm.Blues, ax=ax) # type: ignore

        plt.tight_layout()
        plt.savefig(ML_RESULT_DIR + f"/{self.param_dict['clf_name']}/confusion_matrix.{self.param_dict['clf_name']}_{self.split_mode_name}.png")
        plt.close()

    def get_params(self) -> dict:
        return self.param_dict

    def save_model(self, model_path: str | None = None) -> None:
        self.logger.debug("SAVE", extra={"addinfo": "学習結果の保存\n\n"})
        if model_path is None:
            model_path = ML_MODEL_DIR + f"/model/{self.param_dict['split_mode']}/model_{self.param_dict['clf_name']}_{self.param_dict['training_parameter']}_{self.param_dict['split_mode']}.{dict_to_str(self.param_dict['tuning_params'])}.sav"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            f.close()


if __name__ == "__main__":
    import json
    import sys

    from config.params import SRC_PATH, VARIABLE_PARAMETERS_FOR_TRAINING
    from config.SetLogger import logger_conf

    logger = logger_conf("ML")
    logger.debug("START", extra={"addinfo": "処理開始"})

    # 基本情報
    clf_name = "kNeighbors"  # kNeighbors, LinearSVC, rbfSVC, XGBoost
    # split_mode = input("split_mode : ") # sep, mixsep, mix
    split_mode = "sep"  # sep, mixsep, mix
    if split_mode == "sep":
        split_mode_label = int(input("split_mode_label : "))
        split_mode_name = split_mode + str(split_mode_label)
    else:
        split_mode_label = None
        split_mode_name = split_mode

    # 学習用パラメータ設定
    with open(SRC_PATH + "/config/tuning_parameter.json", "r", encoding="utf-8") as f:
        ML_TUNING_PARAM_DICT = json.load(f)
        tuning_parameter = ML_TUNING_PARAM_DICT[clf_name][split_mode_name]

    with open(SRC_PATH + "/config/fixed_parameter.json", "r", encoding="utf-8") as f:
        ML_FIXED_PARAM_DICT = json.load(f)
        fixed_params = ML_FIXED_PARAM_DICT[clf_name]

    pca = False
    test_size = 0.3
    model_random_state = 42

    # 新規モデルの学習
    method = "training"  # training, predict
    if method == "training":
        for training_parameter in VARIABLE_PARAMETERS_FOR_TRAINING:  # ["density", "energy", "enstrophy", "pressure", "magfieldx", "magfieldy", "velocityx", "velocityy"]
            logger.debug("PARAMETER", extra={"addinfo": f"name={clf_name}, split_mode={split_mode_name}, training_parameter={training_parameter}, pca={pca}, test_size={test_size}, random_state={model_random_state}"})
            model = SupervisedML.load_npys(split_mode=split_mode, training_parameter=training_parameter, split_mode_label=split_mode_label, pca=pca, test_size=test_size, random_state=model_random_state)
            tuning_params = tuning_parameter[training_parameter]
            model.do_learning(clf_name=clf_name, tuning_params=tuning_params, fixed_params=fixed_params)
            model.predict()
            model.print_scores()
            # path = ML_MODEL_DIR + f"/model/{split_mode}/model_{clf_name}_{training_parameter}_{split_mode_name}.optuna.sav"
            model.save_model()

    elif method == "predict":
        for training_parameter in VARIABLE_PARAMETERS_FOR_TRAINING:
            logger.debug("LOAD", extra={"addinfo": f"モデルの読み込み (name={clf_name}, split_mode={split_mode_name}, training_parameter={training_parameter})"})
            tuning_params = tuning_parameter[training_parameter]
            # path = ML_MODEL_DIR + f"/model/{split_mode}/model_{clf_name}_{training_parameter}_{split_mode}.split_mode_label={split_mode_label}.C={param_dict['C']}.sav"
            model = SupervisedML.load_model(training_parameter, split_mode=split_mode, clf_name=clf_name, model_random_state=model_random_state, split_mode_label=split_mode_label)
            model.predict()
            model.print_scores()

    logger.debug("END", extra={"addinfo": "処理終了\n"})
