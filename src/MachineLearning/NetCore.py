# -*- coding: utf-8 -*-

import os
import sys

from torch import nn
import torch.nn.functional as F  # noqa: N812

sys.path.append(os.getcwd() + "/src")

from config.params import ROOT_DIR


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.relu = nn.ReLU()

        # Pooling層:（領域のサイズ, ストライド）
        self.pool = nn.MaxPool2d(2, 2)  # kernel_size=(2,2), stride=2

        # 畳み込み層:(入力チャンネル数, フィルタ数、フィルタサイズ)
        self.conv1 = nn.Conv2d(3, 6, 5)  # C_in=3, C_out=6, kernel_size=(5,5)
        self.conv2 = nn.Conv2d(6, 16, 3)  # C_in=6, C_out=16, kernel_size=(3,3)

        # 全結合層
        self.fc1 = nn.Linear(16 * 23 * 23, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        """
        順伝播の定義
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # 全結合層に入れるため、バッチ内の各画像データをそれぞれ一列にする
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
