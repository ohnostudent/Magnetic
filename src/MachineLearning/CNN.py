# -*- coding: utf-8 -*-

import os
import sys
from logging import getLogger

import matplotlib.pyplot as plt
import torch
import torchvision
from torch import cuda, nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

sys.path.append(os.getcwd() + "/src")

from src.config.params import ML_DATA_DIR
from src.MachineLearning.NetCore import Net


BATCH_SIZE = 3025

class CnnTrain:
    logger = getLogger("ML").getChild("CNN")

    def __init__(self) -> None:
        pass

    def set_train(self):
        data_transform = {"train": transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]), "test": transforms.Compose([transforms.ToTensor()])}
        all_data_set = torchvision.datasets.ImageFolder(root=ML_DATA_DIR + "/cnn/", transform=data_transform["train"])

        # 学習データ、検証データに 7:3 の割合で分割する。
        train_size = int(0.7 * len(all_data_set))
        val_size = len(all_data_set) - train_size
        train_dataset, val_dataset = random_split(all_data_set, [train_size, val_size])

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        self.test_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        return self.train_loader, self.test_loader

    def set_net(self) -> Net:
        net = Net()
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.net = net.to(self.device)

        return self.net

    def set_some(self, WEIGHT_DECAY: float = 0.005, LEARNING_RATE: float = 0.0001):
        # 損失関数の設定
        self.criterion = nn.CrossEntropyLoss()

        # 最適化手法を設定
        # optimizer = optim.Adam(net.parameters())
        self.optimizer = optim.SGD(self.net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

        return self.criterion, self.optimizer

    def run_epoch(self, loader, train_phase):
        sum_loss, sum_correct, sum_total = [0.0, 0, 0]
        dataset_size = len(loader)
        for inputs, labels in loader:
            inputs_gpu, labels_gpu = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs_gpu)
            loss = self.criterion(outputs, labels_gpu)
            if train_phase:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            sum_loss += loss.item()  # lossを足していく
            _, predicted = outputs.max(1)  # 出力の最大値の添字(予想位置)を取得
            sum_total += labels_gpu.size(0)  # labelの数を足していくことでデータの総和を取る
            sum_correct += (predicted == labels_gpu).sum().item()  # 予想位置と実際の正解を比べ,正解している数だけ足す

        # 1エポック分の評価値を計算
        e_loss = sum_loss / dataset_size  # 1エポックの平均損失
        e_acc = sum_correct.double() / dataset_size  # 1エポックの正解率
        print(f"({'train' if train_phase else 'val'}) Loss: {e_loss:.4f} Acc: {e_acc:.4f}")

        return e_loss, e_acc

    def run(self, do_plot):
        train_loss_value = []  # trainingのlossを保持するlist
        train_acc_value = []  # trainingのaccuracyを保持するlist
        val_loss_value = []  # trainingのlossを保持するlist
        val_acc_value = []  # trainingのaccuracyを保持するlist
        test_loss_value = []  # testのlossを保持するlist
        test_acc_value = []  # testのaccuracyを保持するlist

        EPOCH = 100
        train_phase = True
        for epoch in range(EPOCH):
            print("epoch", epoch + 1)  # epoch数の出力

            e_loss, e_acc = self.run_epoch(self.train_loader, train_phase)
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
            self.plot(train_loss_value, train_acc_value, test_loss_value, test_acc_value, EPOCH)

    def plot(self, train_loss_value, train_acc_value, test_loss_value, test_acc_value, EPOCH):
        plt.figure(figsize=(6, 6))  # グラフ描画用

        # 以下グラフ描画
        plt.plot(range(EPOCH), train_loss_value)
        plt.plot(range(EPOCH), test_loss_value, c="#00ff00")
        plt.xlim(0, EPOCH)
        plt.xlabel("EPOCH")
        plt.ylabel("LOSS")
        plt.legend(["train loss", "test loss"])
        plt.title("loss")
        plt.savefig("loss_image.png")
        plt.clf()

        plt.plot(range(EPOCH), train_acc_value)
        plt.plot(range(EPOCH), test_acc_value, c="#00ff00")
        plt.xlim(0, EPOCH)
        plt.xlabel("EPOCH")
        plt.ylabel("ACCURACY")
        plt.legend(["train acc", "test acc"])
        plt.title("accuracy")
        plt.savefig("accuracy_image.png")
