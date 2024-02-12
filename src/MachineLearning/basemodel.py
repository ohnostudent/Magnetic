# -*- coding utf-8, LF -*-

import os
import sys
from glob import glob
from logging import getLogger

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd() + "/src")

from config.params import DATASETS, ML_DATA_DIR, ML_MODEL_DIR, TRAIN_SHAPE, dict_to_str


class BaseModel:
    def __init__(self, training_parameter: str | None = None) -> None:
        self.logger = getLogger("basemodel").getChild("BaseModel")
        # self.set_default(training_parameter) # 初期化

    def set_default(self, training_parameter: str) -> None:
        """初期化処理

        Args:
            training_parameter (str): 使用するパラメータ

        """
        self.param_dict: dict = dict()
        self.param_dict["split_mode"] = None
        self.param_dict["model_name"] = None
        self.param_dict["training_parameter"] = training_parameter
        self.param_dict["dataset_params"] = dict()
        self.param_dict["dataset_params"]["pca"] = False
        self.param_dict["tuning_params"] = dict()
        self.param_dict["fixed_params"] = dict()

    @classmethod
    def load_npys(cls, split_mode: str, training_parameter: str, random_state: int | None = 42, split_mode_label: int | None = None, test_size: float = 0.3, pca: bool = False):  # noqa: ANN206
        """
        データの読み込みを行う関数

        Args:
            split_mode (str): 分割手法
            training_parameter (str): 使用するパラメータ
            random_state (int | None, optional): 乱数の seed値. Defaults to 100.
            test_size (float, optional): 訓練データと検証データの比率. Defaults to 0.3.
            pca (bool, optional): 次元削減を行うか. Defaults to False.

        Returns:
            _type_: モデル
        """
        model = cls(training_parameter)
        model.logger.debug("LOAD", extra={"addinfo": "データの読み込み"})
        model.logger.debug("START", extra={"addinfo": f"training_parameter={training_parameter}, split_mode={split_mode}"})

        model.set_default(training_parameter)  # 初期化
        model.param_dict["training_parameter"] = training_parameter
        model.param_dict["split_mode"] = split_mode
        model.param_dict["split_mode_label"] = split_mode_label
        model.param_dict["dataset_params"]["pca"] = pca
        model.param_dict["dataset_params"]["test_size"] = test_size
        model.param_dict["dataset_params"]["random_state"] = random_state
        model.split_mode_name = split_mode + str(split_mode_label) if split_mode == "sep" else split_mode

        if split_mode == "all":
            train_test_data = np.load(ML_MODEL_DIR + f"/npz/{training_parameter}_all.npz")
        else:
            # ./ML/models/npz/density_mixsep.random_state=100.test_size=0.3.npz
            mode_name = split_mode + str(split_mode_label) if split_mode == "sep" else split_mode
            npz_path = ML_MODEL_DIR + f"/npz/{training_parameter}_{mode_name}.pca={pca}.randomstate={random_state}.testsize={test_size}.npz"
            train_test_data = np.load(npz_path)
            model.X_test = train_test_data["X_test"]
            model.y_test = train_test_data["y_test"]

        model.X_train = train_test_data["X_train"]
        model.y_train = train_test_data["y_train"]

        return model

    # TODO overload
    def _load_npys(self, split_mode: str, training_parameter: str, split_mode_label: int | None = None, random_state: int | None = 42, test_size: float = 0.3, pca: bool = False):
        """学習用モデルを読み込む

        Args:
            training_parameter (str): 使用パラメータ
            split_mode (str): 教師データの分割方法
            split_mode_label (int | None, optional): テストデータに用いるデータセットの指定(mode=sep の場合のみ). [0(77). 1(497), 2(4949)]. Defaults to None.
            test_size (float, optional): 教師データ分割の割合. Defaults to 0.3.
            pca (bool, optional): 次元削減を行うかどうか. Defaults to False.
            random_state (int | None, optional): 分割に用いる乱数. Defaults to 42.
        """

        # パラメータの保存
        self.logger.debug("START", extra={"addinfo": f"parameter={training_parameter}, split_mode={split_mode}"})
        self.param_dict["split_mode"] = split_mode
        self.param_dict["parameter"] = training_parameter
        if split_mode_label is not None:
            self.param_dict["split_mode_label"] = split_mode_label

        self.param_dict["dataset_params"]["pca"] = pca
        self.param_dict["dataset_params"]["random_state"] = random_state
        self.param_dict["dataset_params"]["test_size"] = test_size

        # モデルのロード
        self.split_mode_name = split_mode + str(split_mode_label) if split_mode == "sep" else split_mode
        path = ML_MODEL_DIR + f"/npz/{training_parameter}_{self.split_mode_name}.pca={pca}.randomstate={random_state}.testsize={test_size}.npz"
        train_test_data = np.load(path)
        # 学習用データの振り分け
        self.X_train = train_test_data["X_train"]
        self.y_train = train_test_data["y_train"]
        self.X_test = train_test_data["X_test"]
        self.y_test = train_test_data["y_test"]

    def split_train_test(self, split_mode: str, test_size: float = 0.3, random_state: int | None = 42, split_mode_label: int = 0, pca: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        教師用データとテスト用データを分割する関数

        Args:
            split_mode (str): 分割手法 (sep, mixsep, mix)
            test_size (float, optional): テスト用データの割合. Defaults to 0.3.
            random_state (int | None, optional): 乱数. Defaults to 42.
            split_mode_label (int, optional): ラベル. Defaults to 1.
            pca (bool, optional): 次元削減を行うか否か. Defaults to False.

        Raises:
            ValueError

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        self.path_n, self.path_x, self.path_o = self._load_file_path()  # フォルダパスの取得
        self.split_mode_name = split_mode + str(split_mode_label) if split_mode == "sep" else split_mode

        self.param_dict["split_mode"] = split_mode
        self.param_dict["mode_name"] = self.split_mode_name
        self.param_dict["dataset_params"]["testsize"] = test_size
        self.param_dict["dataset_params"]["randomstate"] = random_state

        if split_mode == "all":
            train_paths, train_label = self._get_train_all()
            self.X_train, self.y_train = self._set_data(train_paths, train_label)
            return self.X_train, self.y_train

        else:
            if split_mode == "mix":
                train_paths, test_paths, train_label, test_label = self._split_train_test_mix(test_size=test_size, random_state=random_state)
            elif split_mode == "mixsep":
                train_paths, test_paths, train_label, test_label = self._split_train_test_sep_mix(test_size=test_size, random_state=random_state)
            elif split_mode == "sep":
                self.param_dict["dataset_params"]["split_mode_label"] = split_mode_label
                train_paths, test_paths, train_label, test_label = self._split_train_test_sep(split_mode_label)
            else:
                raise ValueError(f"Mode '{split_mode}' is not supported.")

            self.X_train, self.y_train = self._set_data(train_paths, train_label)
            self.X_test, self.y_test = self._set_data(test_paths, test_label)

            if pca:
                self.exePCA()

            return self.X_train, self.y_train, self.X_test, self.y_test

    def _get_train_all(self) -> tuple[list[str], list[int]]:
        """
        全データの読み込み

        Args:
            save (bool, optional): 保存. Defaults to True.

        Returns:
            ndarray: バイナリデータ
            ndarray: ラベルデータ

        """
        self.path_n, self.path_x, self.path_o = self._load_file_path()
        path_n_all: list = sum(self.path_n, [])
        path_x_all: list = sum(self.path_x, [])
        path_o_all: list = sum(self.path_o, [])

        train_paths: list = path_n_all.copy()
        train_paths.extend(path_x_all)
        train_paths.extend(path_o_all)

        train_label = [0] * len(path_n_all)
        train_label.extend([1] * len(path_o_all))
        train_label.extend([2] * len(path_x_all))
        return train_paths, train_label

    def dilute(self, ALT_IMAGES: str) -> None:
        """
        上下、左右反転の画像を作成し、教師データをかさましする

        Args:
            ALT_IMAGES (str):  保存先のパス
        """
        # リコネクションがない画像ファイルのパスのリストを取得
        self._save_altImage(self.y_train, ALT_IMAGES)
        # リコネクションがある画像ファイルのパスのリストを取得
        self._save_altImage(self.X_train, ALT_IMAGES)

    def exePCA(self, N_dim: int = 100, randomstate: int | None = 42) -> None:
        """
        次元削減を行う関数

        Args:
            N_dim (int, optional): _description_. Defaults to 100.
            randomstate (int | None, optional): _description_. Defaults to 100.
        """
        self.param_dict["dataset_params"]["pca"] = True
        pca = PCA(n_components=N_dim, random_state=randomstate)

        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)
        self.y_train = pca.fit_transform(self.y_train)
        self.y_test = pca.transform(self.y_test)

        print("PCA累積寄与率: {0}".format(sum(pca.explained_variance_ratio_)))

    def save_npys(self):
        """
        データの保存
        """

        if self.split_mode_name == "all":
            np.savez_compressed(ML_MODEL_DIR + f"/npz/{self.param_dict['training_parameter']}_{split_mode}.npz", X_train=self.X_train, y_train=self.y_train)

        else:
            np.savez_compressed(
                ML_MODEL_DIR + f"/npz/{self.param_dict['training_parameter']}_{self.param_dict['split_mode']}.{dict_to_str(self.param_dict['dataset_params'])}.npz",
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
            )

    def _load_file_path(self) -> tuple[list[list], list[list], list[list]]:
        """
        使用する全ファイルパスの取得

        Returns:
            list[list], list[list], list[list]: ファイルパスのリスト
        """
        path_n, path_x, path_o = list(), list(), list()
        for dataset in DATASETS:
            path_n.append(glob(ML_DATA_DIR + f"/snap_files/{self.param_dict['training_parameter']}/point_n/snap{dataset}*.npy"))
            path_o.append(glob(ML_DATA_DIR + f"/snap_files/{self.param_dict['training_parameter']}/point_o/snap{dataset}*.npy"))
            path_x.append(glob(ML_DATA_DIR + f"/snap_files/{self.param_dict['training_parameter']}/point_x/snap{dataset}*.npy"))

        return path_n, path_x, path_o

    def _load_snap_data(self, file_path: str) -> np.ndarray:
        """
        データのロードを行う関数

        Arg:
            file_path (str) : ファイルパス

        Returns:
            ndarray : 読み込んだデータをnumpy配列として読み込む

        """

        # データのインポート
        extension = file_path[-4:]
        # if os.path.splitext(file_path)[1] == "npy": # 拡張子の判定
        if extension == ".npy":
            img_binary = np.load(file_path)

        elif extension == ".npz":
            print("npz does not supported")
            raise ValueError

        else:
            # r : 読み込み, b : バイナリモード
            with open(file_path, split_mode="rb") as f:  # type: ignore
                img_binary = np.fromfile(f, dtype="f", sep="")
                f.close()

        return img_binary.flatten()

    def _split_train_test_sep(self, test_label: int) -> tuple[list[str], list[str], list[int], list[int]]:
        train_label = {0: 77, 1: 497, 2: 4949}
        train_label.pop(test_label)
        a, b = train_label.keys()

        train_paths = self.path_n[a] + self.path_x[a] + self.path_o[a] + self.path_n[b] + self.path_x[b] + self.path_o[b]
        test_paths = self.path_n[test_label] + self.path_x[test_label] + self.path_o[test_label]

        train_labels = [0] * (len(self.path_n[a]) + len(self.path_n[b]))
        train_labels.extend([1] * (len(self.path_x[a]) + len(self.path_x[b])))
        train_labels.extend([2] * (len(self.path_o[a]) + len(self.path_o[b])))

        test_labels = [0] * len(self.path_n[test_label])
        test_labels.extend([1] * len(self.path_x[test_label]))
        test_labels.extend([2] * len(self.path_o[test_label]))

        return train_paths, test_paths, train_labels, test_labels

    def _split_train_test_sep_mix(self, test_size: float = 0.3, random_state: int | None = 42) -> tuple[list[str], list[str], list[int], list[int]]:
        path_n_all: list = sum(self.path_n, [])
        path_x_all: list = sum(self.path_x, [])
        path_o_all: list = sum(self.path_o, [])

        assert len(path_n_all) != 0
        assert len(path_x_all) != 0
        assert len(path_o_all) != 0

        train_n, test_n = train_test_split(path_n_all, test_size=test_size, random_state=random_state)
        train_x, test_x = train_test_split(path_x_all, test_size=test_size, random_state=random_state)
        train_o, test_o = train_test_split(path_o_all, test_size=test_size, random_state=random_state)

        train_paths: list = train_n.copy()
        train_paths.extend(train_o)
        train_paths.extend(train_x)

        test_paths: list = test_n.copy()
        test_paths.extend(test_o)
        test_paths.extend(test_x)

        train_label = [0] * len(train_n)
        train_label.extend([1] * len(train_x))
        train_label.extend([2] * len(train_o))

        test_label = [0] * len(test_n)
        test_label.extend([1] * len(test_x))
        test_label.extend([2] * len(test_o))

        return train_paths, test_paths, train_label, test_label

    def _split_train_test_mix(self, test_size: float = 0.3, random_state: int | None = 42) -> tuple[list[str], list[str], list[int], list[int]]:
        path_n_all: list = sum(self.path_n, [])
        path_x_all: list = sum(self.path_x, [])
        path_o_all: list = sum(self.path_o, [])
        path_all = path_n_all.copy()
        path_all.extend(path_x_all)
        path_all.extend(path_o_all)

        labels_all = [0] * len(path_n_all)
        labels_all.extend([1] * len(path_x_all))
        labels_all.extend([2] * len(path_o_all))

        train_paths, test_paths, train_label, test_label = train_test_split(path_all, labels_all, test_size=test_size, random_state=random_state)
        return train_paths, test_paths, train_label, test_label

    def _set_data(self, ml_data, labels) -> tuple[np.ndarray, np.ndarray]:
        X_data = np.zeros((len(ml_data), np.prod(TRAIN_SHAPE)))
        label_data = np.array(labels)

        for idx, d in enumerate(ml_data):
            X_data[idx, :] = self._load_snap_data(d)

        return X_data, label_data

    def _alt_array_save(self, item: str, out_path: str) -> None:
        img = np.load(item)
        file_name = os.path.basename(item)

        img_flip = np.flipud(img)  # 画像の上下反転
        print(out_path + "flip_" + file_name, img_flip)
        # np.save(temp_output_dir + "flip_" + file_name , img_flip) # 画像保存

        img_mirror = np.fliplr(img)  # 画像の左右反転
        # np.save(temp_output_dir + "mirr_" + file_name , img_mirror) # 画像保存

        # img_T = mf.resize(img.T, IMGSHAPE) # 画像の上下左右反転
        # np.save(temp_output_dir + "trns_" + file_name , img_T) # 画像保存

    def _save_altImage(self, files: np.ndarray, ALT_IMAGES: str) -> None:
        if not os.path.exists(ALT_IMAGES):
            raise ValueError("IMAGES is not correct")

        out_path = ALT_IMAGES + f"/{self.param_dict['training_parameter']}"
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for file in files:
            self._alt_array_save(file, out_path)


if __name__ == "__main__":
    from config.params import VARIABLE_PARAMETERS_FOR_TRAINING
    from config.SetLogger import logger_conf

    logger = logger_conf("basemodel")
    split_mode = "mix"  # "mix", "mixsep", "sep", "all"
    split_mode_label = 0  # 0, 1, 2
    logger.debug("PARAMETER", extra={"addinfo": f"split_mode = {split_mode}"})

    bm = BaseModel()
    for training_parameter in VARIABLE_PARAMETERS_FOR_TRAINING:
        logger.debug("START", extra={"addinfo": f"{training_parameter}"})

        bm.set_default(training_parameter)
        bm.split_train_test(split_mode)
        # bm.save_npys()

        logger.debug("END", extra={"addinfo": f"{training_parameter}"})

    logger.debug("END", extra={"addinfo": f"split_mode = {split_mode}"})
