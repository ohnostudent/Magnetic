# -*- coding: utf-8 -*-

import torch.nn.functional as F  # noqa: N812
from torch import nn, optim


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.relu = nn.ReLU()

        # Pooling層:（領域のサイズ, ストライド）
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size=(2,2), stride=2

        # 畳み込み層:(入力チャンネル数, フィルタ数、フィルタサイズ)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1, stride=1)  # C_in=3, C_out=6, kernel_size=(5,5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1, stride=1)  # C_in=6, C_out=16, kernel_size=(3,3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=0, stride=1)  # C_in=6, C_out=16, kernel_size=(3,3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.5)

        # 全結合層
        self.fc1 = nn.Linear(64 * 11 * 11, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

        self.set_criterion()
        self.set_optimizer()

    def forward(self, x):
        """
        順伝播の定義
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.conv3(x)))

        # 全結合層に入れるため、バッチ内の各画像データをそれぞれ一列にする
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = self.dropout1(x)
        x = self.fc4(x)

        return F.softmax(x, dim=1)

    def set_optimizer(self, WEIGHT_DECAY: float = 0.0005, LEARNING_RATE: float = 0.0001):
        # 最適化手法を設定
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # self.optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.001, weight_decay=WEIGHT_DECAY)
        return self.optimizer

    def set_criterion(self):
        # 損失関数の設定
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion
