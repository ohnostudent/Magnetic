# -*- coding utf-8, LF -*-

import os
import pickle
import sys
from glob import glob

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd() + "/src")

from config.params import DATASETS, IMAGE_SHAPE, LABELS, ML_DATA_DIR, ML_MODEL_DIR  # noqa: E402


class BaseModel:
    def __init__(self, parameter: str) -> None:
        self.PARAMETER = parameter
        self.path_n, self.path_x, self.path_o = self._load_file_path()
        self.mode = None

    @classmethod
    def load_model(cls, path, parameter):  # noqa: ANN206
        model = cls(parameter)
        model.model = pickle.load(open(path, "rb"))
        return model

    @classmethod
    def load_npys(cls, mode, parameter):  # noqa: ANN206
        train_test_data = np.load(ML_MODEL_DIR + f"/npz/{mode}_{parameter}.npz")
        X_train, y_train, X_test, y_test = train_test_data

        model = cls(parameter)
        model.mode = mode
        return X_train, y_train, X_test, y_test

    def split_train_test(self, mode, label=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if mode == "sep":
            train_paths, self.test_paths, train_label, test_label = self._split_train_test_sep(label)

        elif mode == "mixsep":
            train_paths, self.test_paths, train_label, test_label = self._split_train_test_sep_mix()

        elif mode == "mix":
            train_paths, self.test_paths, train_label, test_label = self._split_train_test_mix()

        else:
            raise ValueError(f"Mode '{mode}' is not supported.")

        self.X_train, self.y_train = self._set_data(train_paths, train_label)
        self.X_test, self.y_test = self._set_data(self.test_paths, test_label)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def dilute(self, ALT_IMAGES):
        """
        上下、左右反転の画像を作成し、教師データをかさましする
        """
        # リコネクションがない画像ファイルのパスのリストを取得
        self._save_altImage(self.y_train, ALT_IMAGES)
        # リコネクションがある画像ファイルのパスのリストを取得
        self._save_altImage(self.X_train, ALT_IMAGES)

    def exePCA(self, N_dim=100, randomstate=None):
        pca = PCA(n_components=N_dim, random_state=randomstate)

        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

        print(f"PCA累積寄与率: {sum(pca.explained_variance_ratio_)}")

    def save_npys(self, X_train, y_train, X_test, y_test):
        np.savez_compressed(ML_MODEL_DIR + f"/npz/{self.PARAMETER}_{self.mode}.npz", X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    def save_model(self, model_name) -> None:
        model_path = ML_MODEL_DIR + f"/{model_name}/model_{model_name}_{self.PARAMETER}_{self.mode}"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            f.close()

    def _load_file_path(self):
        path_n, path_x, path_o = list(), list(), list()
        for dataset in DATASETS:
            path_n.append(glob(ML_DATA_DIR + f"/snap_files/snap{dataset}/point_n/{self.PARAMETER}/*"))
            path_o.append(glob(ML_DATA_DIR + f"/snap_files/snap{dataset}/point_o/{self.PARAMETER}/*"))
            path_x.append(glob(ML_DATA_DIR + f"/snap_files/snap{dataset}/point_x/{self.PARAMETER}/*"))

        return path_n, path_x, path_o

    def _load_snap_data(self, file_path: str) -> np.ndarray:
        """
        データのロードを行う関数

        Arg:
            file_path (str) : ファイルパス
            z (int) :

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
            with open(file_path, mode="rb") as f:
                img_binary = np.fromfile(f, dtype="f", sep="")
                f.close()

        return img_binary.flatten()

    def _split_train_test_sep(self, test_label):
        train_label = [0, 1, 2]
        train_label.remove(test_label)
        a, b = train_label

        train_paths = self.path_n[a] + self.path_x[a] + self.path_o[a] + self.path_n[b] + self.path_x[b] + self.path_o[b]
        test_paths = self.path_n[test_label] + self.path_x[test_label] + self.path_o[test_label]

        train_labels = [0] * (len(self.path_n[a]) + len(self.path_n[b]))
        train_labels.extend([1] * (len(self.path_x[a]) + len(self.path_x[b])))
        train_labels.extend([2] * (len(self.path_o[a]) + len(self.path_o[b])))

        test_labels = [0] * len(self.path_n[test_label])
        test_labels.extend([1] * len(self.path_x[test_label]))
        test_labels.extend([2] * len(self.path_o[test_label]))

        return train_paths, test_paths, train_labels, test_labels

    def _split_train_test_sep_mix(self, test_size=0.3, random_state=100):
        path_n = sum(self.path_n, [])
        path_x = sum(self.path_x, [])
        path_o = sum(self.path_o, [])

        train_n, test_n = train_test_split(path_n, test_size=test_size, random_state=random_state)
        train_x, test_x = train_test_split(path_x, test_size=test_size, random_state=random_state)
        train_o, test_o = train_test_split(path_o, test_size=test_size, random_state=random_state)

        train_paths: list = train_n
        train_paths.extend(train_o)
        train_paths.extend(train_x)

        test_paths: list = test_n
        test_paths.extend(test_o)
        test_paths.extend(test_x)

        train_label = [0] * len(train_n)
        train_label.extend([1] * len(train_x))
        train_label.extend([2] * len(train_o))

        test_label = [0] * len(test_n)
        test_label.extend([1] * len(test_x))
        test_label.extend([2] * len(test_o))

        return train_paths, test_paths, train_label, test_label

    def _split_train_test_mix(self, test_size=0.3, random_state=100):
        path_all = self.path_n
        path_all.extend(self.path_o)
        path_all.extend(self.path_x)

        labels_all = [0] * len(self.path_n)
        labels_all.extend([1] * len(self.path_x))
        labels_all.extend([2] * len(self.path_o))

        train_paths, test_paths, train_label, test_label = train_test_split(path_all, LABELS, test_size=test_size, random_state=random_state)
        return train_paths, test_paths, train_label, test_label

    def _set_data(self, ml_data, labels) -> tuple[np.ndarray, np.ndarray]:
        X_data = np.zeros((len(ml_data), np.prod(IMAGE_SHAPE)))
        label_data = np.array(labels)

        for idx, d in enumerate(ml_data):
            X_data[idx, :] = self._load_snap_data(d)

        return X_data, label_data

    def _alt_array_save(self, item, out_path) -> None:
        img = np.load(item)
        file_name = os.path.basename(item)

        img_flip = np.flipud(img)  # 画像の上下反転
        print(out_path + "flip_" + file_name, img_flip)
        # np.save(temp_output_dir + "flip_" + file_name , img_flip) # 画像保存

        img_mirror = np.fliplr(img)  # 画像の左右反転
        # np.save(temp_output_dir + "mirr_" + file_name , img_mirror) # 画像保存

        # img_T = mf.resize(img.T, IMGSHAPE) # 画像の上下左右反転
        # np.save(temp_output_dir + "trns_" + file_name , img_T) # 画像保存

    def _save_altImage(self, files, ALT_IMAGES):
        if not os.path.exists(ALT_IMAGES):
            raise ValueError("IMAGES is not correct")

        out_path = ALT_IMAGES + f"/{self.PARAMETER}"
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for file in files:
            self._alt_array_save(file, out_path)

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model
