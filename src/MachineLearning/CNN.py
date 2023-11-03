# -*- coding: utf-8 -*-

import os
import sys
from logging import getLogger

import matplotlib.pyplot as plt
import torch
from torch import Generator, cuda, device
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.append(os.getcwd() + "/src")

from config.params import ML_DATA_DIR, ML_MODEL_DIR, ML_RESULT_DIR
from MachineLearning.NetCore import Net


class CnnTrain:
    logger = getLogger("CNN").getChild("Train")

    def __init__(self) -> None:
        pass

    def set_train(self, seed: int = 42):
        data_transform = {
            "train": transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
            "test": transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]),
        }
        all_data_set = ImageFolder(root=ML_DATA_DIR + "/cnn", transform=data_transform["train"])

        # 学習データ、検証データに 7:3 の割合で分割
        train_size = 0.5
        val_size = 0.2
        test_size = 0.3
        train_dataset, val_dataset, test_dataset = random_split(all_data_set, [train_size, val_size, test_size], generator=Generator().manual_seed(seed))

        BATCH_SIZE = 1024
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        return self.train_loader, self.val_loader, self.test_loader

    def set_net(self) -> Net:
        net = Net()
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        return self.net

    def _run_epoch(self, loader, train_phase):
        sum_loss = 0.0  # lossの合計
        sum_correct = 0  # 正解率の合計
        sum_total = 0  # dataの数の合計

        for inputs, labels in loader:
            inputs_gpu, labels_gpu = inputs.to(self.device), labels.to(self.device)  # GPU にデータを移行
            outputs = self.net(inputs_gpu)  # 推論を実施（順伝播による出力）
            loss = self.net.criterion(outputs, labels_gpu)  # 交差エントロピーによる損失計算（バッチ平均値

            if train_phase:
                self.net.optimizer.zero_grad()  # 誤差逆伝播の勾配をリセット
                loss.backward()  # 誤差逆伝播
                self.net.optimizer.step()  # パラメータ更新

            sum_loss += loss.item()  # lossを足していく
            predicted = torch.argmax(outputs, dim=1)  # 出力の最大値の添字(予想位置)を取得
            sum_total += labels_gpu.size(0)  # labelの数を足していくことでデータの総和を取る
            sum_correct += predicted.eq(labels_gpu.view_as(predicted)).sum().item()  # 予想位置と実際の正解を比べ,正解している数だけ足す

        # 1エポック分の評価値を計算
        e_loss = sum_loss / len(loader)  # 1エポックの平均損失
        e_acc = float(sum_correct / sum_total)  # 1エポックの正解率
        self.logger.debug("SCORE", extra={"addinfo": f"({'train' if train_phase else 'val'}) Loss: {e_loss:.4f} Acc: {e_acc:.4f}, correct: {sum_correct}, total: {sum_total}"})

        return e_loss, e_acc

    def run(self, epoch_cnt: int = 5, do_plot: bool = True):
        # cuda.empty_cache()  # GPUのリセット
        train_loss_value = []  # trainingのlossを保持するlist
        train_acc_value = []  # trainingのaccuracyを保持するlist
        val_loss_value = []  # validationのlossを保持するlist
        val_acc_value = []  # validationのaccuracyを保持するlist
        test_loss_value = []  # testのlossを保持するlist
        test_acc_value = []  # testのaccuracyを保持するlist

        self.logger.debug("START", extra={"addinfo": ""})
        for epoch in range(epoch_cnt):
            self.logger.debug("epoch", extra={"addinfo": f"{epoch + 1 :03d}"})

            self.net.train(True)
            e_loss, e_acc = self._run_epoch(self.train_loader, train_phase=True)
            train_loss_value.append(e_loss)
            train_acc_value.append(e_acc)

            # 検証フェーズ
            self.net.train(False)
            with torch.no_grad():  # 無駄に勾配計算しないように
                e_loss, e_acc = self._run_epoch(self.val_loader, train_phase=False)
            val_loss_value.append(e_loss)
            val_acc_value.append(e_acc)

            # テストフェーズ
            self.net.train(False)
            with torch.no_grad():  # 無駄に勾配計算しないように
                e_loss, e_acc = self._run_epoch(self.test_loader, train_phase=False)
            test_loss_value.append(e_loss)
            test_acc_value.append(e_acc)

        if do_plot:
            self.logger.debug("PLOT", extra={"addinfo": "グラフ保存"})
            self.plot(train_loss_value, train_acc_value, val_loss_value, val_acc_value, test_loss_value, test_acc_value, EPOCH)

        self.logger.debug("END", extra={"addinfo": ""})

    def plot(self, train_loss_value, train_acc_value, val_loss_value, val_acc_value, test_loss_value, test_acc_value, EPOCH: int = 100):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        ax_loss = axes[0]
        ax_loss.plot(range(EPOCH), train_loss_value)
        ax_loss.plot(range(EPOCH), val_loss_value)
        ax_loss.plot(range(EPOCH), test_loss_value, c="#00ff00")
        ax_loss.set_xlim(0, EPOCH)
        ax_loss.set_xlabel("EPOCH")
        ax_loss.set_ylabel("LOSS")
        ax_loss.legend(["train loss", "validation loss", "test loss"])
        ax_loss.set_title("loss")

        ax_acc = axes[1]
        ax_acc.plot(range(EPOCH), train_acc_value)
        ax_acc.plot(range(EPOCH), val_acc_value)
        ax_acc.plot(range(EPOCH), test_acc_value, c="#00ff00")
        ax_acc.set_xlim(0, EPOCH)
        ax_acc.set_xlabel("EPOCH")
        ax_acc.set_ylabel("ACCURACY")
        ax_acc.legend(["train acc", "validation acc", "test acc"])
        ax_acc.set_title("accuracy")

        plt.tight_layout()
        plt.savefig(ML_RESULT_DIR + "/cnn/loss_acc_image.png")
        plt.show()

    def save_model(self, weight_only: bool = False, save_path: str | None = None):
        if save_path is None:
            save_path = ML_MODEL_DIR + "/model/cnn_{mode}_cpu.pth"

        if weight_only:
            if "{mode}" in save_path:
                save_path.format(mode="weight")
            save_model = self.net.state_dict()
        else:
            if "{mode}" in save_path:
                save_path.format(mode="model")
            save_model = self.net

        torch.save(save_model, save_path)

    @classmethod
    def load_model(cls, mode: str = "model", path: str | None = None):  # noqa: ANN206
        if path is None:
            path = ML_MODEL_DIR + "/model/cnn_cpu.pth"

        model = cls()
        model.device = device("cuda" if cuda.is_available() else "cpu")

        if mode == "wight":
            net = Net()
            net.load_state_dict(torch.load(path, map_location="cpu"))
        elif mode == "model":
            net = torch.load(path, map_location="cpu")
        else:
            raise ValueError()

        model.net = net.to(model.device)
        return model


if __name__ == "__main__":
    # from config.params import VARIABLE_PARAMETERS_FOR_TRAINING
    from config.SetLogger import logger_conf

    logger = logger_conf("CNN")
    EPOCH = 100

    model = CnnTrain()
    model.set_net()
    model.set_train(seed=100)
    model.run(epoch_cnt=EPOCH, do_plot=False)
    model.save_model()
