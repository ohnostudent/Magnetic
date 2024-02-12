# -*- coding utf-8, LF -*-

"""
CNN の実行

"""

import os
import sys
from datetime import datetime
from glob import glob
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import Generator, cuda, device, no_grad
from torch import argmax as torch_argmax, load as torch_load, save as torch_save
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder

sys.path.append(os.getcwd() + "/src")

from config.params import LABELS, ML_DATA_DIR, ML_MODEL_DIR, ML_RESULT_DIR
from MachineLearning.NetCore import Net


class LoadDataset(Dataset):
    def __init__(self, dataset_list: list, parameter: str | None = None, extension: str = "npy", transform: transforms.Compose | None = None) -> None:
        self.dataset_list = dataset_list
        self.extension = extension
        self.transform = transform

        if extension == "npy":
            self.load_method = self._bin_loader
            self.path = ML_DATA_DIR + f"/snap_files/{parameter}"
        elif extension == "bmp":
            self.load_method = self._pil_loader
            self.path = ML_DATA_DIR + "/cnn"
        else:
            raise ValueError("invalid extension")

        self.img_path_and_label = self._load()

    def __getitem__(self, index):  # noqa: ANN204
        img_path, split_mode_label = self.img_path_and_label[index]
        img = self.load_method(img_path)

        if self.transform:
            img = self.transform(img)

        return img, int(split_mode_label)

    def __len__(self) -> int:
        return len(self.img_path_and_label)

    def _load(self) -> np.ndarray:
        all_dataset = list()
        for dataset in self.dataset_list:
            all_dataset.append(self._load_dataset(dataset))

        img_path_and_label = np.concatenate(all_dataset, 0)
        np.random.shuffle(img_path_and_label)
        return img_path_and_label

    def _load_dataset(self, dataset) -> np.ndarray:
        all_dataset = list()
        for split_mode_label in LABELS:
            all_dataset.append(self._load_image_path(dataset, split_mode_label))

        img_path_label_for_dataset = np.concatenate(all_dataset, 0)
        np.random.shuffle(img_path_label_for_dataset)
        return img_path_label_for_dataset

    def _load_image_path(self, snap_dataset, split_mode_label: str) -> np.ndarray:
        data_path = np.array(glob(self.path + f"/point_{split_mode_label}/snap{snap_dataset}_*.{self.extension}"))
        labels = np.full(len(data_path), LABELS.index(split_mode_label), dtype=np.int8)
        img_path_and_label = np.stack([data_path, labels], 1)
        np.random.shuffle(img_path_and_label)
        return img_path_and_label

    def _pil_loader(self, path: str):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _bin_loader(self, path: str):
        return np.load(path)


class CnnTrain:
    def __init__(self, training_parameter: str, split_mode: str, split_mode_label: int | None = None) -> None:
        """_summary_

        Args:
            training_parameter (str): _description_
            split_mode (str): _description_
            split_mode_label (int | None, optional): _description_. Defaults to None.
        """
        self.logger = getLogger("CNN").getChild("Train")
        self.training_parameter = training_parameter
        self.split_mode = split_mode

        if split_mode == "sep":
            self.split_mode_label = split_mode_label
            self.split_mode_name = split_mode + str(split_mode_label)
        else:
            self.split_mode_name = split_mode

        if training_parameter == "images":
            self.extension = "bmp"
        else:
            self.extension = "npy"
        self.logger.debug("PARAMETER", extra={"addinfo": f"parameter={training_parameter}, split_mode={self.split_mode_name}, extension={self.extension}"})

    @classmethod
    def load_model(cls, training_parameter: str, split_mode: str, split_mode_label: int | None = None, load_mode: str = "model", load_path: str | None = None):
        if load_path is None:
            load_path = ML_MODEL_DIR + "/model/cnn_cpu.pth"

        if load_mode == "wight":
            net = Net()
            net.load_state_dict(torch_load(load_path, map_location="cpu"))
        elif load_mode == "model":
            net = torch_load(load_path, map_location="cpu")
        else:
            raise ValueError()

        model = cls(training_parameter, split_mode, split_mode_label)
        model.device = device("cuda" if cuda.is_available() else "cpu")
        model.net = net.to(model.device)
        model.logger.debug("MODEL PATH", extra={"addinfo": f"{load_path}"})
        model.logger.debug("Device", extra={"addinfo": f"{model.device}"})
        model.logger.debug("NET", extra={"addinfo": f"\n{net}"})
        return model

    def _set_channel(self) -> int:
        if self.extension == "bmp":
            channel = 1
        elif self.extension == "npy":
            if self.training_parameter == "mag_tuple":
                channel = 2
            else:
                channel = 1
        else:
            raise ValueError()
        return channel

    def set_net(self) -> Net:
        cuda.empty_cache()  # GPUのリセット

        channel = self._set_channel()
        net = Net(self.extension, channel=channel)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.net = net.to(self.device)

        self.logger.debug("Device", extra={"addinfo": f"{self.device}"})
        self.logger.debug("NET", extra={"addinfo": f"\n{net}"})
        return self.net

    def configure_transform(self) -> None:
        if self.extension == "bmp":
            self.data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        elif self.extension == "npy":
            self.data_transform = transforms.Compose([transforms.ToTensor()])
        else:
            raise ValueError()

    def set_train(self, BATCH_SIZE=512, seed: int = 42) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_size = 0.6
        val_size = 0.2
        test_size = 0.2
        generator = Generator().manual_seed(seed)
        self.configure_transform()

        if self.split_mode == "mix":
            # 学習データ、検証データ、テストデータに 6:2:2 の割合で分割
            all_data_set = self._use_folder()
            train_dataset, val_dataset, test_dataset = random_split(all_data_set, [train_size, val_size, test_size], generator=generator)

        elif self.split_mode == "mixsep":
            train_dataset_0, val_dataset_0, test_dataset_0 = random_split(LoadDataset([77], self.training_parameter, self.extension, self.data_transform), [train_size, val_size, test_size], generator=generator)
            train_dataset_1, val_dataset_1, test_dataset_1 = random_split(LoadDataset([497], self.training_parameter, self.extension, self.data_transform), [train_size, val_size, test_size], generator=generator)
            train_dataset_2, val_dataset_2, test_dataset_2 = random_split(LoadDataset([4949], self.training_parameter, self.extension, self.data_transform), [train_size, val_size, test_size], generator=generator)
            train_dataset = train_dataset_0 + train_dataset_1 + train_dataset_2
            val_dataset = val_dataset_0 + val_dataset_1 + val_dataset_2
            test_dataset = test_dataset_0 + test_dataset_1 + test_dataset_2

        elif self.split_mode == "sep":
            if self.split_mode_label == 0:
                train_list = [497, 4949]
                val_list = [77]
            elif self.split_mode_label == 1:
                train_list = [77, 4949]
                val_list = [497]
            elif self.split_mode_label == 2:
                train_list = [77, 497]
                val_list = [4949]
            else:
                raise ValueError("invalid split mode")

            train_dataset: Dataset = LoadDataset(train_list, self.training_parameter, self.extension, self.data_transform)
            val_test_dataset: Dataset = LoadDataset(val_list, self.training_parameter, self.extension, self.data_transform)
            val_dataset, test_dataset = random_split(val_test_dataset, [0.5, 0.5], generator=generator)

        else:
            raise ValueError()

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        return self.train_loader, self.val_loader, self.test_loader

    def run(self, epoch_cnt: int = 5, do_plot: bool = True) -> None:
        train_loss_value = []  # training の loss を保持する list
        train_acc_value = []  # training の accuracy を保持する list
        val_loss_value = []  # validation の loss を保持する list
        val_acc_value = []  # validation の accuracy を保持する list
        test_loss_value = []  # test の loss を保持する list
        test_acc_value = []  # test の accuracy を保持する list

        self.logger.debug("START", extra={"addinfo": ""})
        for epoch in range(epoch_cnt):
            self.logger.debug("epoch", extra={"addinfo": f"{epoch + 1 :03d}"})

            # 学習の実行
            e_loss, e_acc = self._run_epoch(self.train_loader, phase="train")
            train_loss_value.append(e_loss)
            train_acc_value.append(e_acc)

            # 検証
            with no_grad():  # 無駄に勾配計算しないように
                e_loss, e_acc = self._run_epoch(self.val_loader, phase="validation")
            val_loss_value.append(e_loss)
            val_acc_value.append(e_acc)

            # テスト
            with no_grad():  # 無駄に勾配計算しないように
                e_loss, e_acc = self._run_epoch(self.test_loader, phase="test")
            test_loss_value.append(e_loss)
            test_acc_value.append(e_acc)

        if do_plot:  # 結果のグラフ化
            self.logger.debug("PLOT", extra={"addinfo": "グラフ保存"})
            self.plot(train_loss_value, train_acc_value, val_loss_value, val_acc_value, test_loss_value, test_acc_value)

        self.logger.debug("END", extra={"addinfo": ""})

    def predict(self) -> None:
        self.logger.debug("PREDICT", extra={"addinfo": ""})
        self.pred_proba = []
        self.y_test = []

        for inputs, labels in self.test_loader:
            with no_grad():
                inputs_gpu, labels_gpu = inputs.to(self.device), labels.to(self.device)  # GPU にデータを移行
                output = self.net(inputs_gpu)  # 推論を実施(順伝播による出力)

            # 結果の保存
            self.pred_proba.extend([int(i.argmax()) for i in output])
            self.y_test.extend([int(i) for i in labels_gpu])

        self.print_scores()

    def print_scores(self) -> None:
        """
        評価データの出力
        """
        save_path = ML_RESULT_DIR + f"/cnn/{self.split_mode_name}_{self.training_parameter}.txt"
        f = open(save_path, "a", encoding="utf-8")
        self.logger.debug("SCORE", extra={"addinfo": f"path={save_path}"})

        time = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
        print(f"【 {time} 】", file=f)

        print("変数       :", self.training_parameter, self.split_mode, file=f)
        acc_score = accuracy_score(self.y_test, self.pred_proba)
        print("Accuracy   :", acc_score, file=f)
        # auc_score = roc_auc_score(self.y_test, self.pred_proba, multi_class="ovr")
        # print("AUC        :", auc_score, file=f)

        print("適合率macro:", precision_score(self.y_test, self.pred_proba, average="macro"), file=f)
        print("再現率macro:", recall_score(self.y_test, self.pred_proba, average="macro"), file=f)
        f1_macro = f1_score(self.y_test, self.pred_proba, average="macro")
        print("F1値  macro:", f1_macro, file=f)

        print("適合率micro:", precision_score(self.y_test, self.pred_proba, average="micro"), file=f)
        print("再現率micro:", recall_score(self.y_test, self.pred_proba, average="micro"), file=f)
        print("F1値  micro:", f1_score(self.y_test, self.pred_proba, average="micro"), file=f)

        clf_rep = confusion_matrix(self.y_test, self.pred_proba)
        self._plot_confusion_matrix(clf_rep)
        print("混合行列   : \n", clf_rep, file=f)
        print("要約       : \n", classification_report(self.y_test, self.pred_proba, digits=3), file=f)

        print("\n\n\n", file=f)
        self.logger.debug("SCORE", extra={"addinfo": f"acc_score={acc_score}, f1_macro={f1_macro}"})
        f.close()

    def _plot_confusion_matrix(self, clf_rep) -> None:
        fig, ax = plt.subplots(1, 1)
        cmp = ConfusionMatrixDisplay(clf_rep)  # type: ignore
        ax.set_title(f"CNN_{self.training_parameter}_{self.split_mode_name}")
        cmp.plot(cmap=plt.cm.Blues, ax=ax)  # type: ignore

        plt.tight_layout()
        plt.savefig(ML_RESULT_DIR + f"/cnn/confusion_matrix.CNN_{self.split_mode_name}.png")
        plt.close()

    def plot(self, train_loss_value, train_acc_value, val_loss_value, val_acc_value, test_loss_value, test_acc_value) -> None:
        """
        学習結果をプロット
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 4 * 2))
        epoch_num = len(train_acc_value)

        # 正解率
        ax_acc = axes[0]
        ax_acc.plot(range(epoch_num), train_acc_value)
        ax_acc.plot(range(epoch_num), val_acc_value)
        ax_acc.plot(range(epoch_num), test_acc_value, c="#00ff00")
        ax_acc.set_xlim(-epoch_num // 10, epoch_num * 1.1)
        ax_acc.set_xlabel("EPOCH", fontsize=12)
        ax_acc.set_ylabel("ACCURACY", fontsize=12)
        ax_acc.tick_params(labelsize=14)
        ax_acc.legend(["train acc", "validation acc", "test acc"])
        ax_acc.set_title("Accuracy", fontsize=15)

        # 損失
        ax_loss = axes[1]
        ax_loss.plot(range(epoch_num), train_loss_value)
        ax_loss.plot(range(epoch_num), val_loss_value)
        ax_loss.plot(range(epoch_num), test_loss_value, c="#00ff00")
        ax_loss.set_xlim(-epoch_num // 10, epoch_num * 1.1)
        ax_loss.set_xlabel("EPOCH", fontsize=12)
        ax_loss.set_ylabel("LOSS", fontsize=12)
        ax_loss.tick_params(labelsize=14)
        ax_loss.legend(["train loss", "validation loss", "test loss"])
        ax_loss.set_title("Loss", fontsize=15)

        plt.tight_layout()
        plt.savefig(ML_RESULT_DIR + f"/cnn/loss_acc_image.{self.extension}_{self.training_parameter}_{self.split_mode_name}.png")
        plt.close()

    def save_model(self, weight_only: bool = False, save_path: str | None = None) -> None:
        # 保存パスの作成
        if weight_only:  # 重みのみの保存
            split_mode = "weight"
            save_model = self.net.state_dict()
        else:
            split_mode = "model"
            save_model = self.net

        if save_path is None:
            save_path = ML_MODEL_DIR + f"/model/{self.split_mode}/model_cnn_{self.extension}_{self.training_parameter}_{self.split_mode_name}.save={split_mode}.pth"

        self.logger.debug("SAVE", extra={"addinfo": f"{save_path}"})
        # 保存
        torch_save(save_model, save_path)

    def _npy_loader(self, path) -> np.ndarray:
        return np.load(path)

    def _use_folder(self) -> DatasetFolder | ImageFolder:
        if self.extension == "npy":  # 元データで学習
            if self.training_parameter is None:
                raise ValueError("training_parameter not defined")
            all_data_set = DatasetFolder(root=ML_DATA_DIR + f"/snap_files/{self.training_parameter}", loader=self._npy_loader, extensions=(".npy",), transform=self.data_transform)

        elif self.extension == "bmp":  # 画像データで学習
            all_data_set = ImageFolder(root=ML_DATA_DIR + "/cnn", transform=self.data_transform)

        else:
            raise ValueError("File not found.")

        return all_data_set

    def _run_epoch(self, loader, phase: str = "train") -> tuple[float, float]:
        sum_loss = 0.0  # lossの合計
        sum_correct = 0  # 正解率の合計
        sum_total = 0  # dataの数の合計

        for inputs, labels in loader:
            inputs_gpu, labels_gpu = inputs.to(self.device), labels.to(self.device)  # GPU にデータを移行
            outputs = self.net(inputs_gpu)  # 推論を実施(順伝播)
            loss = self.net.criterion(outputs, labels_gpu)  # 交差エントロピーによる損失計算(バッチ平均値)

            if phase == "train":
                self.net.optimizer.zero_grad()  # 誤差逆伝播の勾配をリセット
                loss.backward()  # 誤差逆伝播
                self.net.optimizer.step()  # パラメータ更新

            sum_loss += loss.item()  # lossを足していく
            predicted = torch_argmax(outputs, dim=1)  # 出力の最大値の添字(予想位置)を取得
            sum_total += labels_gpu.size(0)  # split_mode_labelの数を足していくことでデータの総和を取るa
            sum_correct += predicted.eq(labels_gpu.view_as(predicted)).sum().item()  # 予想位置と実際の正解を比べ,正解している数だけ足す

        # 1エポック分の評価値を計算
        e_loss = sum_loss / len(loader)  # 1エポックの平均損失cz
        e_acc = float(sum_correct / sum_total)  # 1エポックの正解率

        self.logger.debug(f"{phase}", extra={"addinfo": f"Loss: {e_loss}, Accuracy: {e_acc}, correct: {sum_correct}, total: {sum_total}, sumLoss: {sum_loss}"})

        return e_loss, e_acc


if __name__ == "__main__":
    from config.params import VARIABLE_PARAMETERS_FOR_TRAINING
    from config.SetLogger import logger_conf

    logger = logger_conf("CNN")

    # split_mode = input("split_mode : ") # "mixsep"  # sep, mixsep, mix
    split_mode = "mix"  # mix, mixsep, sep

    if split_mode == "sep":
        split_mode_label = int(input("split_mode_label : "))
        split_mode_name = split_mode + str(split_mode_label)
    else:
        split_mode_label = 0
        split_mode_name = split_mode

    process_mode = "train"  # train, predict
    # process_mode = "predict"  # train, predict

    # training_parameter = "density"  # density, energy, enstrophy, pressure, magfieldx, magfieldy, velocityx, velocityy, images
    if process_mode == "train":
        EPOCH = 100
        for training_parameter in VARIABLE_PARAMETERS_FOR_TRAINING:
            model = CnnTrain(training_parameter=training_parameter, split_mode=split_mode, split_mode_label=split_mode_label)
            model.set_net()
            model.set_train(seed=42)
            model.run(epoch_cnt=EPOCH, do_plot=True)
            model.predict()
            model.save_model()

    elif process_mode == "predict":
        for training_parameter in VARIABLE_PARAMETERS_FOR_TRAINING:
            path = ML_MODEL_DIR + f"/model/{split_mode}/model_cnn_npy_{training_parameter}_{split_mode_name}.save=model.device=cuda.pth"
            model = CnnTrain.load_model(training_parameter=training_parameter, split_mode=split_mode, split_mode_label=split_mode_label, load_path=path)
            model.set_train(seed=42)
            model.predict()
