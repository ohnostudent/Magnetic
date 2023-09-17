# -*- coding utf-8, LF -*-

import os
import pickle
import sys
from glob import glob

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/src")

from config.params import ML_DATA_DIR, ML_MODEL_DIR, datasets, labels  # noqa: E402


class BaseModel:
    def __init__(self, parameter: str, imgshape=(100, 25), model_name=None) -> None:
        self.PARAMETER = parameter
        self.MODEL_NAME = model_name
        self.JUDGE_COLUMN = "is_reconnecting"  # (0,1)
        self.IMGSHAPE = imgshape  # 出来れば画像サイズはすべて同じで合ってほしい。違うサイズが混じる場合は最も多いサイズを指定する
        self.path_n, self.path_x, self.path_o = self._load_file_path()

    @classmethod
    def load_model(cls, path, parameter, imgshape=(100, 25), model_name=None):  # noqa: ANN206
        model = cls(parameter, imgshape=imgshape, model_name=model_name)

        model.model = pickle.load(open(path, "rb"))
        return model

    def _load_file_path(self):
        path_n, path_x, path_o = list(), list(), list()
        for dataset in datasets:
            path_n.extend(glob(ML_DATA_DIR + f"/snap_files/snap{dataset}/point_n/{self.PARAMETER}/*"))
            path_o.extend(glob(ML_DATA_DIR + f"/snap_files/snap{dataset}/point_o/{self.PARAMETER}/*"))
            path_x.extend(glob(ML_DATA_DIR + f"/snap_files/snap{dataset}/point_x/{self.PARAMETER}/*"))

        return path_n, path_x, path_o

    def split_train_test(self, mode) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if mode == "sep":
            # train_data, test_data, train_label, test_label = self._split_train_test_sep()
            raise ValueError("Mode 'sep' is not supported yet.")

        elif mode == "mixsep":
            train_data, test_data, train_label, test_label = self._split_train_test_sep_mix()

        elif mode == "mix":
            train_data, test_data, train_label, test_label = self._split_train_test_mix()

        else:
            raise ValueError(f"Mode '{mode}' is not supported.")

        self.X_train, self.y_train = self._set_data(train_data, train_label)
        self.X_test, self.y_test = self._set_data(test_data, test_label)

        return self.X_train, self.y_train, self.X_test, self.y_test

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

    def dilute(self, ALT_IMAGES):
        """
        上下、左右反転の画像を作成し、教師データをかさましする
        """
        # リコネクションがない画像ファイルのパスのリストを取得
        self._save_altImage(self.y_train, ALT_IMAGES)
        # リコネクションがある画像ファイルのパスのリストを取得
        self._save_altImage(self.X_train, ALT_IMAGES)

    def exePCA(self, randomstate=None):
        N_dim = 100  # 100列に落とし込む
        pca = PCA(n_components=N_dim, random_state=randomstate)

        self.X_train = pca.fit_transform(self.X_train)
        self.X_test = pca.transform(self.X_test)

        print("PCA累積寄与率: {0}".format(sum(pca.explained_variance_ratio_)))

    def print_score(self, models):
        print("Train :", models[0].score(self.X_train, self.y_train))
        print("Test :", models[0].score(self.X_test, self.y_test))
        print(models[2])

    def save_model(self, prt=False) -> None:
        model_path = ML_MODEL_DIR + f"/{self.MODEL_NAME}/model_{self.PARAMETER}_{self.model}"

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            f.close()

        if prt:
            self.print_score(self.model)

    def _loadSnapData(self, file_path: str, z=3) -> np.ndarray:
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

        elif extension == ".jpg":
            img_binary = Image.open(file_path).convert("L")
            img_binary = img_binary.resize(self.IMGSHAPE)  # 画像のサイズ変更
            img_binary = np.ravel(np.array(img_binary))  # 画像を配列に変換

        else:
            # r : 読み込み, b : バイナリモード
            with open(file_path, mode="rb") as f:
                if z == 1:
                    img_binary = np.fromfile(f, dtype="f", sep="").reshape(1025, 513)
                elif z == 3:
                    img_binary = np.fromfile(f, dtype="f", sep="")[:525825].reshape(1025, 513)
                else:
                    raise ValueError

                f.close()
        return img_binary

    def _resize(self, array: np.ndarray, yx: tuple[int, int]):
        if array.shape == yx:
            return array

        print("resized:", array.shape, "to", self.IMGSHAPE)
        im = Image.fromarray(array, mode="L")
        im = im.resize(yx, resample=Image.LANCZOS)
        return np.array(im)

    def _load_regularize(self, data):
        img = self._loadSnapData(data)  # データの読み込み
        img_resize = self._resize(img, self.IMGSHAPE)  # リサイズ処理

        normalization = self._normalization(img_resize)  # 正規化
        return normalization

    def _normalization(self, img_resize: np.ndarray) -> np.ndarray:
        return ((img_resize - min(img_resize.flatten())) / (max(img_resize.flatten()) - min(img_resize.flatten()))).flat

    # def _split_train_test_sep(self, test_size=0.3, randomstate=100):
    #     if self.TARGET == "x":
    #         train_data = self.path_n
    #     return train_data, test_data, train_label, test_label

    def _split_train_test_sep_mix(self, test_size=0.3, random_state=100):
        train_n, test_n = train_test_split(self.path_n, test_size=test_size, random_state=random_state)
        train_o, test_o = train_test_split(self.path_o, test_size=test_size, random_state=random_state)
        train_x, test_x = train_test_split(self.path_x, test_size=test_size, random_state=random_state)

        train_data: list = train_n
        train_data.extend(train_o)
        train_data.extend(train_x)

        test_data: list = test_n
        test_data.extend(test_o)
        test_data.extend(test_x)

        train_label = [0] * len(train_n)
        train_label.extend([1] * len(train_x))
        train_label.extend([2] * len(train_o))

        test_label = [0] * len(test_n)
        test_label.extend([1] * len(test_x))
        test_label.extend([2] * len(test_o))

        return train_data, test_data, train_label, test_label

    def _split_train_test_mix(self, test_size=0.3, random_state=100):
        path_all = self.path_n
        path_all.extend(self.path_o)
        path_all.extend(self.path_x)

        labels_all = [0] * len(self.path_n)
        labels_all.extend([1] * len(self.path_x))
        labels_all.extend([2] * len(self.path_o))

        train_data, test_data, train_label, test_label = train_test_split(path_all, labels, test_size=test_size, random_state=random_state)
        return train_data, test_data, train_label, test_label

    def _set_data(self, ml_data, labels) -> tuple[np.ndarray, np.ndarray]:
        X_data = np.zeros((len(ml_data), np.prod(self.IMGSHAPE)))
        label_data = np.zeros(len(ml_data))

        for idx, d in enumerate(zip(ml_data, labels, strict=True)):
            X_data[idx, :] = self._load_regularize(d[0])
            label_data[idx] = d[1]

        return X_data, label_data

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model
