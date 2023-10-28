# -*- coding utf-8, LF -*-

import os
import pickle
import sys
from datetime import datetime
from logging import getLogger

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from torch import cuda
from xgboost import XGBClassifier

sys.path.append(os.getcwd() + "/src")

from config.params import ML_MODEL_DIR, ML_RESULT_DIR, dict_to_str
from MachineLearning.basemodel import BaseModel


class SupervisedML(BaseModel):
    logger = getLogger("ML").getChild("Machine Learning")
    CUDA = cuda.is_available()
    if CUDA:
        cuda.empty_cache()

    def __init__(self, parameter: str) -> None:
        super().__init__(parameter)
        self.set_default(parameter)

    @classmethod
    def load_model(cls, parameter: str, param_dict: dict, mode: str = "mixsep", name: str = "LinearSVC", model_random_state: int = 100, path: str | None = None):  # noqa: ANN206
        model = cls(parameter)
        model.param_dict["mode"] = mode
        model.param_dict["parameter"] = parameter
        model.param_dict["model_name"] = name

        if path is None:
            path = ML_MODEL_DIR + f"/model/model_{name}_{parameter}_{mode}.{dict_to_str(param_dict['train_params'])}.sav"
            model.param_dict["clf_params"] = param_dict
        else:
            # TODO 小数に対応
            path_params = path.split(".")[1:]
            for param in path_params:
                key, val = param.split("=")
                model.param_dict[key] = val

        model._load_npys(mode=mode, parameter=parameter, random_state=model_random_state)
        model.model = pickle.load(open(path, "rb"))
        return model

    def do_learning(self, clf_name: str, param_dict: dict):
        """学習を行う関数

        Args:
            clf_name (str): 使用する分類器の名前 (kneighbors / linearSVC / rbfSVC / XGBoost)
            param_dict(dict): 各種パラメータを格納する辞書, 各値は以下の通り.
            - kneighbors
                - n_clusters            : クラスターの個数
                - init                  : セントロイドの初期値をランダムに設定  default: 'k-means++'
                - n_init                : 異なるセントロイドの初期値を用いたk-meansの実行回数 default: '10' 実行したうちもっとSSE値が小さいモデルを最終モデルとして選択
                - max_iter              : k-meansアルゴリズムの内部の最大イテレーション回数  default: '300'
                - tol                   : 収束と判定するための相対的な許容誤差 default: '1e-04'
                - random_state          : セントロイドの初期化に用いる乱数発生器の状態

            - linearSVC
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
        logger.debug("START", extra={"addinfo": f"学習開始 ({clf_name})"})
        match clf_name:
            case "KMeans":
                self.model = self.KMeans(**param_dict)
            case "kneighbors":
                self.model = self.kneighbors(**param_dict)
            case "linearSVC":
                self.model = self.linearSVC(**param_dict)
            case "rbfSVC":
                self.model = self.rbfSVC(**param_dict)
            case "XGBoost":
                self.model = self.XGBoost(**param_dict)
        logger.debug("END", extra={"addinfo": "学習終了"})
        return self.model

    def KMeans(self, n_clusters: int = 3, n_init: int = 10, max_iter: int = 300, tol: float = 1e-04, random_state: int = 100) -> KMeans:
        self.logger.debug("START", extra={"addinfo": "学習開始"})
        self.param_dict["model_name"] = "LinearSVC"
        self.param_dict["clf_params"]["n_clusters"] = n_clusters
        self.param_dict["clf_params"]["n_init"] = n_init
        self.param_dict["clf_params"]["max_iter"] = max_iter
        self.param_dict["clf_params"]["tol"] = tol
        self.param_dict["clf_params"]["random_state"] = random_state

        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
        self.model.fit(self.X_train)

        return self.model

    def kneighbors(self, n_neighbors: int | None = 3) -> KNeighborsClassifier:
        self.param_dict["model_name"] = "KNeighbors"
        self.param_dict["clf_params"]["n_neighbors"] = n_neighbors

        if n_neighbors is None:
            n_neighbors = int(np.sqrt(self.X_train.shape[0]))  # kの設定

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def linearSVC(self, C: float = 0.3, random_state: int = 0) -> LinearSVC:
        self.param_dict["model_name"] = "LinearSVC"
        self.param_dict["clf_params"]["C"] = C
        self.param_dict["clf_params"]["randomstate"] = random_state

        self.model = LinearSVC(C=C, random_state=random_state)  # インスタンスを生成
        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def rbfSVC(self, C: float = 0.3, gamma: float = 3, random_state: int = 100, cls: str = "ovo") -> OneVsOneClassifier | OneVsRestClassifier:
        self.param_dict["model_name"] = f"rbfSVC-{cls}"
        self.param_dict["clf_params"]["C"] = C
        self.param_dict["clf_params"]["gamma"] = gamma
        self.param_dict["clf_params"]["randomstate"] = random_state

        estimator = SVC(C=C, kernel="rbf", gamma=gamma, random_state=random_state)
        match cls:
            # インスタンスを生成
            case "ovo":
                self.model = OneVsOneClassifier(estimator)
            case "ovr":
                self.model = OneVsRestClassifier(estimator)
            case _:
                raise ValueError

        self.model.fit(self.X_train, self.y_train)  # モデルの学習
        return self.model

    def XGBoost(self, colsample_bytree: float = 0.4, early_stopping_rounds: int = 100, eval_metric: str = "auc", learning_rate: float = 0.02, max_depth: int = 4, missing: int = -1, n_estimators: int = 500, subsample: float = 0.8, params: dict | None = None) -> XGBClassifier:
        self.param_dict["model_name"] = "XGBoost"
        self.param_dict["clf_params"]["n_estimators"] = n_estimators
        self.param_dict["clf_params"]["max_depth"] = max_depth
        self.param_dict["clf_params"]["learning_rate"] = learning_rate
        self.param_dict["clf_params"]["subsample"] = subsample
        self.param_dict["clf_params"]["colsample_bytree"] = colsample_bytree
        self.param_dict["clf_params"]["missing"] = missing
        self.param_dict["clf_params"]["eval_metric"] = eval_metric
        self.param_dict["clf_params"]["early_stopping_rounds"] = early_stopping_rounds

        if self.CUDA:
            tree_methods = "gpu_hist"
        else:
            tree_methods = "hist"

        self.param_dict["clf_params"]["tree_methods"] = tree_methods

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            missing=missing,
            eval_metric=eval_metric,
            tree_method=tree_methods,
        )

        if params is not None:
            self.model.set_params(**params)

        eval_set = [(self.X_train, self.y_train)]
        self.model.fit(model.X_train, model.y_train, eval_set=eval_set, verbose=50, early_stopping_rounds=early_stopping_rounds)  # モデルの学習

        return self.model

    def predict(self, pred_path: str | np.ndarray | None = None) -> None:
        """
        予測を行う関数

        Args:
            pred_path (str | np.ndarray | None, optional): 予測を行うデータ. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if pred_path is None:
            self.pred = self.model.predict(self.X_test)

        elif isinstance(pred_path, str):
            self.pred = self.model.predict(np.load(pred_path))

        elif isinstance(pred_path, np.ndarray):
            self.pred = self.model.predict(pred_path)

        else:
            raise ValueError("引数の型が違います")

    def print_scores(self) -> None:
        """
        評価データの出力
        """
        f = open(
            ML_RESULT_DIR + f"/{self.param_dict['model_name']}/{self.param_dict['parameter']}_{self.param_dict['mode']}.txt",
            "a",
            encoding="utf-8",
        )
        time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        print(f"【 {time} 】", file=f)
        print("パラメータ : ", dict_to_str(self.param_dict["clf_params"]), "\n", file=f)
        print("trainスコア: ", self.model.score(self.X_train, self.y_train), file=f)
        print("test スコア: ", self.model.score(self.X_test, self.y_test), file=f)
        print("正解率     : ", accuracy_score(self.y_test, self.pred), file=f)
        print("適合率     : ", precision_score(self.y_test, self.pred, average="macro"), file=f)
        print("再現率     : ", recall_score(self.y_test, self.pred, average="macro"), file=f)
        print("F1値       : ", f1_score(self.y_test, self.pred, average="macro"), file=f)
        print("混合行列   : \n", confusion_matrix(self.y_test, self.pred), file=f)
        print("要約       : \n", classification_report(self.y_test, self.pred), file=f)
        print("\n\n\n", file=f)
        f.close()

    def get_params(self) -> dict:
        return self.param_dict

    def save_model(self, model_path: str | None = None) -> None:
        if model_path is None:
            model_path = ML_MODEL_DIR + f"/model/model_{self.param_dict['model_name']}_{self.param_dict['parameter']}_{self.param_dict['mode']}.{dict_to_str(self.param_dict['clf_params'])}.sav"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            f.close()


if __name__ == "__main__":
    import sys

    # from config.params import ML_PARAM_DICT
    from config.SetLogger import logger_conf

    logger = logger_conf("ML")
    logger.debug("START", extra={"addinfo": "処理開始"})

    mode = "mixsep"
    parameter = "density"

    # 教師データ用パラメータ
    pca = False
    test_size = 0.3
    model_random_state = 100

    clf_name = "XGBoost"  # "kneighbors", "linearSVC", "rbfSVC", "XGBoost"
    logger.debug("PARAMETER", extra={"addinfo": f"name={clf_name}, mode={mode}, parameter={parameter}, pca={pca}, test_size={test_size}, random_state={model_random_state}"})

    # 学習用パラメータ設定
    ML_PARAM_DICT = {
        "KMeans": {"n_clusters": 3, "n_init": 10, "max_iter": 300, "tol": 1e-04, "random_state": 100, "verbose": 10},
        "kneighbors": {"n_clusters": 3, "n_init": 10, "max_iter": 300, "tol": 1e-04, "random_state": 100, "verbose": 10},
        "linearSVC": {"C": 0.3, "random_state": 0, "verbose": 10},
        "rbfSVC": {
            "C": 1.0,  # 正則化パラメータ、マージン
            "cache_size": 200,  # キャッシュサイズ
            "coef0": 0.0,  # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
            "decision_function_shape": "ovr",
            "degree": 3,  # 多項式(poly)カーネルの次数
            "gamma": "scale",  # カーネルの係数、ガウスカーネル(rbf): 1/(n_features * X.var()) と シグモイドカーネル(sigmoid): 1 /n_features
            "kernel": "rbf",  # カーネル('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            "max_iter": -1,  # ソルバー内の反復に対するハード制限
            "probability": False,  # True の場合、予測時に各クラスに属する確率を返す
            "random_state": None,  # 乱数の seed値
            "shrinking": True,  # 縮小ヒューリスティックを使用するかどうか
            "tol": 0.001,  # 停止基準の許容値
            "verbose": 2,  # 詳細な出力を有効化
        },
        "XGBoost": {
            "colsample_bytree": 0.4,
            "early_stopping_rounds": 100,
            "eval_metric": "auc",
            "learning_rate": 0.02,
            "max_depth": 4,
            "missing": -1,
            "n_estimators": 1000,
            "subsample": 0.8,
            "params": {},
            "verbose": 50,
        },
    }
    param_dict = ML_PARAM_DICT[clf_name]

    if len(sys.argv) > 1:
        logger.debug("LOAD", extra={"addinfo": f"モデルの読み込み ({clf_name})"})
        model = SupervisedML.load_model(parameter, mode=mode, name=clf_name, model_random_state=model_random_state, param_dict=param_dict)

        logger.debug("PREDICT", extra={"addinfo": "予測"})
        model.predict()

    else:
        logger.debug("LOAD", extra={"addinfo": "データの読み込み"})
        model = SupervisedML.load_npys(mode=mode, parameter=parameter, pca=pca, test_size=test_size, random_state=model_random_state)

        logger.debug("Learning", extra={"addinfo": f"学習開始 ({clf_name})"})
        model.do_learning(clf_name=clf_name, param_dict=param_dict)

        logger.debug("SAVE", extra={"addinfo": "学習結果の保存"})
        model.save_model()

        logger.debug("PREDICT", extra={"addinfo": "予測"})
        model.predict()

        logger.debug("END", extra={"addinfo": "処理終了"})
