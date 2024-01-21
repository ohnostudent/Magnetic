# -*- coding: utf-8 -*-

import os
import sys
from glob import glob
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import Generator
from torch import argmax as torch_argmax
from torch import cuda, device
from torch import load as torch_load
from torch import no_grad
from torch import save as torch_save
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder

sys.path.append(os.getcwd() + "/src")

from config.params import LABELS, ML_DATA_DIR, ML_MODEL_DIR, ML_RESULT_DIR
from MachineLearning.NetCore import Net


class MyDataset(Dataset):
    def __init__(self, dataset_list: list, parameter: str, extension: str = "npy", transform: transforms.Compose | None = None) -> None:
        self.dataset_list = dataset_list
        self.extension = extension
        self.transform = transform

        if extension == "npy":
            self.load_method = self.bin_loader
            self.path = ML_DATA_DIR + f"/snap_files/{parameter}"
        elif extension == "bmp":
            self.load_method = self.pil_loader
            self.path = ML_DATA_DIR + "/cnn"
        else:
            raise ValueError("invalid extension")

        self.img_path_and_label = self.load()

    def __getitem__(self, index):  # noqa: ANN204
        img_path, label = self.img_path_and_label[index]
        img = self.load_method(img_path)

        if self.transform:
            img = self.transform(img)

        return img, int(label)

    def __len__(self) -> int:
        return len(self.img_path_and_label)

    def load(self) -> np.ndarray:
        all_dataset = list()
        for dataset in self.dataset_list:
            all_dataset.append(self._load_dataset(dataset))

        img_path_and_label = np.concatenate(all_dataset, 0)
        np.random.shuffle(img_path_and_label)
        return img_path_and_label

    def _load_dataset(self, dataset) -> np.ndarray:
        all_dataset = list()
        for label in LABELS:
            all_dataset.append(self._load_image_path(dataset, label))

        img_path_label_for_dataset = np.concatenate(all_dataset, 0)
        np.random.shuffle(img_path_label_for_dataset)
        return img_path_label_for_dataset

    def _load_image_path(self, snap_dataset, label) -> np.ndarray:
        data_path = np.array(glob(self.path + f"/point_{LABELS[label]}/snap{snap_dataset}_*.{self.extension}"))
        labels = np.full(len(data_path), label, dtype=np.int8)
        img_path_and_label = np.stack([data_path, labels], 1)
        np.random.shuffle(img_path_and_label)
        return img_path_and_label

    def pil_loader(self, path: str):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def bin_loader(self, path: str):
        return np.load(path)


class CnnTrain:
    def __init__(self, training_parameter: str, mode: str, label: int | None = None) -> None:
        """_summary_

        Args:
            training_parameter (str): _description_
            mode (str): _description_
            label (int | None, optional): _description_. Defaults to None.
        """
        self.logger = getLogger("CNN").getChild("Train")

        self.training_parameter = training_parameter
        self.mode = mode
        if mode == "sep":
            self.label = label
            self.mode_name = mode + str(label)
        else:
            self.mode_name = mode
        logger.debug("PARAMETER", extra={"addinfo": f"parameter={training_parameter}, mode={self.mode_name}"})

    @classmethod
    def load_model(cls, parameter: str, mode: str, load_mode: str = "model", path: str | None = None):  # noqa: ANN206
        if path is None:
            path = ML_MODEL_DIR + "/model/cnn_cpu.pth"

        model = cls(parameter, mode)
        model.device = device("cuda" if cuda.is_available() else "cpu")

        if load_mode == "wight":
            net = Net()
            net.load_state_dict(torch_load(path, map_location="cpu"))
        elif load_mode == "model":
            net = torch_load(path, map_location="cpu")
        else:
            raise ValueError()

        model.net = net.to(model.device)
        model.logger.debug("Device", extra={"addinfo": f"{model.device}"})
        model.logger.debug("NET", extra={"addinfo": f"{net}"})
        return model

    def _set_channel(self):
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

    def set_net(self, extension: str = "bmp") -> Net:
        cuda.empty_cache()  # GPUのリセット
        self.extension = extension
        logger.debug("PARAMETER", extra={"addinfo": f"extension={self.extension}"})

        channel = self._set_channel()
        net = Net(self.extension, channel=channel)
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.net = net.to(self.device)

        self.logger.debug("Device", extra={"addinfo": f"{self.device}"})
        self.logger.debug("NET", extra={"addinfo": f"{net}"})
        return self.net

    def configure_transform(self):
        if self.extension == "bmp":
            self.data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        elif self.extension == "npy":
            self.data_transform = transforms.Compose([transforms.ToTensor()])
        else:
            raise ValueError()

    def set_train(self, BATCH_SIZE=512, seed: int = 42):
        train_size = 0.6
        val_size = 0.2
        test_size = 0.2
        generator = Generator().manual_seed(seed)
        self.configure_transform()

        if self.mode == "mix":
            # 学習データ、検証データ、テストデータに 6:2:2 の割合で分割
            all_data_set = self._use_folder()
            train_dataset, val_dataset, test_dataset = random_split(all_data_set, [train_size, val_size, test_size], generator=generator)

        elif self.mode == "mixsep":
            train_dataset_0, val_dataset_0, test_dataset_0 = random_split(MyDataset([77], self.training_parameter, self.extension, self.data_transform), [train_size, val_size, test_size], generator=generator)
            train_dataset_1, val_dataset_1, test_dataset_1 = random_split(MyDataset([497], self.training_parameter, self.extension, self.data_transform), [train_size, val_size, test_size], generator=generator)
            train_dataset_2, val_dataset_2, test_dataset_2 = random_split(MyDataset([4949], self.training_parameter, self.extension, self.data_transform), [train_size, val_size, test_size], generator=generator)
            train_dataset = train_dataset_0 + train_dataset_1 + train_dataset_2
            val_dataset = val_dataset_0 + val_dataset_1 + val_dataset_2
            test_dataset = test_dataset_0 + test_dataset_1 + test_dataset_2

        elif self.mode == "sep":
            if label == 0:
                train_list = [497, 4949]
                val_list = [77]
            elif label == 1:
                train_list = [77, 4949]
                val_list = [497]
            elif label == 2:
                train_list = [77, 497]
                val_list = [4949]
            else:
                raise ValueError("invalid split mode")

            train_dataset: Dataset = MyDataset(train_list, self.training_parameter, self.extension, self.data_transform)
            val_test_dataset: Dataset = MyDataset(val_list, self.training_parameter, self.extension, self.data_transform)
            val_dataset, test_dataset = random_split(val_test_dataset, [0.5, 0.5], generator=generator)

        else:
            raise ValueError()

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        return self.train_loader, self.val_loader, self.test_loader

    def run(self, epoch_cnt: int = 5, do_plot: bool = True):
        train_loss_value = []  # trainingのlossを保持するlist
        train_acc_value = []  # trainingのaccuracyを保持するlist
        val_loss_value = []  # validationのlossを保持するlist
        val_acc_value = []  # validationのaccuracyを保持するlist
        test_loss_value = []  # testのlossを保持するlist
        test_acc_value = []  # testのaccuracyを保持するlist

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
            self.plot(train_loss_value, train_acc_value, val_loss_value, val_acc_value, test_loss_value, test_acc_value, EPOCH)

        self.logger.debug("END", extra={"addinfo": ""})

    def plot(self, train_loss_value, train_acc_value, val_loss_value, val_acc_value, test_loss_value, test_acc_value, EPOCH: int = 100):
        fig, axes = plt.subplots(2, 1, figsize=(10, 4*2))

        ax_acc = axes[0]
        ax_acc.plot(range(EPOCH), train_acc_value)
        ax_acc.plot(range(EPOCH), val_acc_value)
        ax_acc.plot(range(EPOCH), test_acc_value, c="#00ff00")
        ax_acc.set_xlim(-EPOCH // 10, EPOCH * 1.1)
        ax_acc.set_xlabel("EPOCH", fontsize=12)
        ax_acc.set_ylabel("ACCURACY", fontsize=12)
        ax_acc.tick_params(labelsize=14)
        ax_acc.legend(["train acc", "validation acc", "test acc"])
        ax_acc.set_title("Accuracy", fontsize=15)

        ax_loss = axes[1]
        ax_loss.plot(range(EPOCH), train_loss_value)
        ax_loss.plot(range(EPOCH), val_loss_value)
        ax_loss.plot(range(EPOCH), test_loss_value, c="#00ff00")
        ax_loss.set_xlim(-EPOCH // 10, EPOCH * 1.1)
        ax_loss.set_xlabel("EPOCH", fontsize=12)
        ax_loss.set_ylabel("LOSS", fontsize=12)
        ax_loss.tick_params(labelsize=14)
        ax_loss.legend(["train loss", "validation loss", "test loss"])
        ax_loss.set_title("Loss", fontsize=15)

        plt.tight_layout()
        plt.savefig(ML_RESULT_DIR + f"/cnn/loss_acc_image.{self.extension}_{self.training_parameter}_{self.mode_name}.png")
        plt.close()

    def save_model(self, weight_only: bool = False, save_path: str | None = None):
        # 保存パスの作成
        if weight_only:  # 重みのみの保存
            mode = "weight"
            save_model = self.net.state_dict()
        else:
            mode = "model"
            save_model = self.net

        if save_path is None:
            save_path = ML_MODEL_DIR + f"/model/{self.mode}/model_cnn_{self.extension}_{self.training_parameter}_{self.mode_name}.save={mode}.device={self.device}.pth"

        self.logger.debug("SAVE", extra={"addinfo": f"{save_path}"})
        # 保存
        torch_save(save_model, save_path)

    def _npy_loader(self, path):
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

    def _run_epoch(self, loader, phase: str = "train"):
        sum_loss = 0.0  # lossの合計
        sum_correct = 0  # 正解率の合計
        sum_total = 0  # dataの数の合計

        for inputs, labels in loader:
            inputs_gpu, labels_gpu = inputs.to(self.device), labels.to(self.device)  # GPU にデータを移行
            outputs = self.net(inputs_gpu)  # 推論を実施（順伝播による出力）
            loss = self.net.criterion(outputs, labels_gpu)  # 交差エントロピーによる損失計算（バッチ平均値

            if phase == "train":
                self.net.optimizer.zero_grad()  # 誤差逆伝播の勾配をリセット
                loss.backward()  # 誤差逆伝播
                self.net.optimizer.step()  # パラメータ更新

            sum_loss += loss.item()  # lossを足していく
            predicted = torch_argmax(outputs, dim=1)  # 出力の最大値の添字(予想位置)を取得
            sum_total += labels_gpu.size(0)  # labelの数を足していくことでデータの総和を取るa
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

    # training_parameter = "density"  # density, energy, enstrophy, pressure, magfieldx, magfieldy, velocityx, velocityy
    mode = "mix"  # sep, mixsep, mix
    if mode == "sep":
        label = 0
        mode_name = mode + str(label)
    else:
        label = None
        mode_name = mode

    extension = "bmp"  # npy, bmp
    EPOCH = 100
    for training_parameter in VARIABLE_PARAMETERS_FOR_TRAINING:
        model = CnnTrain(training_parameter=training_parameter, mode=mode, label=label)
        model.set_net(extension=extension)
        model.set_train(seed=42)
        model.run(epoch_cnt=EPOCH, do_plot=True)
        model.save_model()
