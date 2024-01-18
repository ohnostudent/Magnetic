# -*- coding: utf-8 -*-


import torch.nn.functional as f
from torch import nn, optim


class Net(nn.Module):
    def __init__(self, mode: str = "bmp", channel: int = 1) -> None:
        super(Net, self).__init__()

        # 全結合層
        if mode == "bmp":
            self._for_img(channel=1)

        elif mode == "npy":
            self._for_npy(channel=channel)

        else:
            raise ValueError("invalid extension")

        self._configure_criterion()
        self._configure_optimizer()

    def _for_img(self, channel: int = 1):
        self.layer1 = nn.Sequential(
            # 畳み込み層:(入力チャンネル数, フィルタ数、フィルタサイズ)
            nn.Conv2d(in_channels=channel, out_channels=6, kernel_size=3, padding=1, stride=1),  # C_in=3, C_out=6, kernel_size=(5,5)
            nn.ReLU(),
            # Pooling層:（領域のサイズ, ストライド）
            nn.MaxPool2d(2, 2),  # kernel_size=(2,2), stride=2
            nn.Dropout(p=0.4),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1, stride=1),  # C_in=6, C_out=16, kernel_size=(3,3)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size=(2,2), stride=2
            nn.Dropout(p=0.4),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=0, stride=1),  # C_in=6, C_out=16, kernel_size=(3,3)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size=(2,2), stride=2
            nn.Dropout(p=0.4),
        )
        self.fcs = nn.Sequential(
            nn.Linear(64 * 11 * 11, 1024),  # 96 * 96
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 3),
        )

    def _for_npy(self, channel: int = 1):
        self.layer1 = nn.Sequential(
            # 畳み込み層:(入力チャンネル数, フィルタ数、フィルタサイズ)
            nn.Conv2d(in_channels=channel, out_channels=6, kernel_size=3, padding=1, stride=1),  # C_in=3, C_out=6, kernel_size=(5,5)
            nn.ReLU(),
            # Pooling層:（領域のサイズ, ストライド）
            nn.MaxPool2d(1, 2),  # kernel_size=(2,2), stride=2
            nn.Dropout(p=0.4),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1, stride=1),  # C_in=6, C_out=16, kernel_size=(3,3)
            nn.ReLU(),
            nn.MaxPool2d(1, 2),  # kernel_size=(2,2), stride=2
            nn.Dropout(p=0.4),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=0, stride=1),  # C_in=6, C_out=16, kernel_size=(3,3)
            nn.ReLU(),
            nn.MaxPool2d(1, 2),  # kernel_size=(2,2), stride=2
            nn.Dropout(p=0.4),
        )
        self.fcs = nn.Sequential(
            nn.Linear(64 * 25 * 10, 1024),  # 100 * 10
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        """
        順伝播の定義
        """
        # 畳み込み層
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全結合層に入れるため、バッチ内の各画像データをそれぞれ一列にする
        x = x.view(x.size(0), -1)

        # 全結合層
        x = self.fcs(x)

        return f.softmax(x, dim=1)

    def _configure_optimizer(self, WEIGHT_DECAY: float = 0.005, LEARNING_RATE: float = 0.001):
        # 最適化手法を設定
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # self.optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.001, weight_decay=WEIGHT_DECAY)
        return self.optimizer

    def _configure_criterion(self):
        # 損失関数の設定
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion
