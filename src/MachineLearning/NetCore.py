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
        self.conv1 = nn.Conv2d(3, 6, 5)  # C_in=3, C_out=6, kernel_size=(5,5)
        self.conv2 = nn.Conv2d(6, 16, 3, stride=2)  # C_in=6, C_out=16, kernel_size=(3,3)
        self.conv3 = nn.Conv2d(16, 64, 3)  # C_in=6, C_out=16, kernel_size=(3,3)
        self.dropout = nn.Dropout2d(p=0.2)

        # 全結合層
        self.fc1 = nn.Linear(64 * 10 * 10, 1024)
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 3)

        self.set_criterion()

    def forward(self, x):
        """
        順伝播の定義
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # 全結合層に入れるため、バッチ内の各画像データをそれぞれ一列にする
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def set_optimizer(self, WEIGHT_DECAY: float = 0.005, LEARNING_RATE: float = 0.0001):
        # 最適化手法を設定
        # optimizer = optim.Adam(net.parameters())
        self.optimizer = optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        return self.optimizer


    def set_criterion(self):
        # 損失関数の設定
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion
