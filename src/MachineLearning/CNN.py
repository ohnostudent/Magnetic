# -*- coding: utf-8 -*-

import os
import sys
from logging import getLogger

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import cuda
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

sys.path.append(os.getcwd() + "/src")

from config.params import ML_DATA_DIR, ML_RESULT_DIR
from MachineLearning.NetCore import Net

BATCH_SIZE = 3025


class CnnTrain:
    logger = getLogger("ML").getChild("CNN")

    def __init__(self) -> None:
        pass

    def set_train(self):
        data_transform = {"train": transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]), "test": transforms.Compose([transforms.ToTensor()])}
        all_data_set = torchvision.datasets.ImageFolder(root=ML_DATA_DIR + "/cnn", transform=data_transform["train"])

        # 学習データ、検証データに 7:3 の割合で分割
        train_size = int(0.7 * len(all_data_set))
        val_size = len(all_data_set) - train_size
        train_dataset, val_dataset = random_split(all_data_set, [train_size, val_size])

        BATCH_SIZE = train_size // 8
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.test_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        return self.train_loader, self.test_loader

    def set_net(self) -> Net:
        net = Net()
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.net = net.to(self.device)

        return self.net

    def run_epoch(self, loader, train_phase):
        sum_loss = 0.0  # lossの合計
        sum_correct = 0  # 正解率の合計
        sum_total = 0  # dataの数の合計

        for inputs, labels in loader:
            inputs_gpu, labels_gpu = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs_gpu)  # 推論を実施（順伝播による出力）
            loss = self.net.criterion(outputs, labels_gpu)  # 交差エントロピーによる損失計算（バッチ平均値

            if train_phase:
                self.net.optimizer.zero_grad()  # 誤差逆伝播の勾配をリセット
                loss.backward()  # 誤差逆伝播
                self.net.optimizer.step()  # パラメータ更新

            sum_loss += loss.item()  # lossを足していく
            _, predicted = outputs.max(1)  # 出力の最大値の添字(予想位置)を取得
            sum_total += labels_gpu.size(0)  # labelの数を足していくことでデータの総和を取る
            sum_correct += predicted.eq(labels_gpu.view_as(predicted)).sum().item()  # 予想位置と実際の正解を比べ,正解している数だけ足す

        # 1エポック分の評価値を計算
        dataset_size = len(loader)
        e_loss = sum_loss * BATCH_SIZE / dataset_size  # 1エポックの平均損失
        e_acc = float(sum_correct / sum_total)  # 1エポックの正解率
        self.logger.debug("SCORE", extra={"addinfo": f"({'train' if train_phase else 'val'}) Loss: {e_loss:.4f} Acc: {e_acc:.4f}"})

        return e_loss, e_acc

    def run(self, do_plot):
        cuda.empty_cache()
        train_loss_value = []  # trainingのlossを保持するlist
        train_acc_value = []  # trainingのaccuracyを保持するlist
        val_loss_value = []  # validationのlossを保持するlist
        val_acc_value = []  # validationのaccuracyを保持するlist
        test_loss_value = []  # testのlossを保持するlist
        test_acc_value = []  # testのaccuracyを保持するlist

        EPOCH = 100
        self.logger.debug("START", extra={"addinfo": ""})
        for epoch in range(EPOCH):
            self.logger.debug("epoch", extra={"addinfo": f"{epoch + 1}"})

            e_loss, e_acc = self.run_epoch(self.train_loader, train_phase=True)
            train_loss_value.append(e_loss)
            train_acc_value.append(e_acc)

            # 検証フェーズ
            with torch.no_grad():  # 無駄に勾配計算しないように
                e_loss, e_acc = self.run_epoch(self.train_loader, train_phase=False)
            val_loss_value.append(e_loss)
            val_acc_value.append(e_acc)

            # テストフェーズ
            with torch.no_grad():  # 無駄に勾配計算しないように
                e_loss, e_acc = self.run_epoch(self.test_loader, train_phase=False)
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
