# 磁気リコネクション

<right>written by Kuniya Ota</right>

23 卒の先輩方が行った研究のデータをもとに、リファクタリングを行ったもの

※1 /Magnetic を作業ディレクトリとしてください

```
> pwd
./Magnetic
```

※2 場合によっては以下のコードを追記してください

```python
import os
import sys
sys.path.append(os.getcwd() + "/src")
```

<br>

## 1. 元データ

1. コマンドラインにて以下のコードを実行する

   ```
   pip install -r requirements.txt
   ```

2. `/cmd/mkdir.bat` を実行する

3. 元データを `./data` 配下に保存する

   ```
   /data/ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0
   ```

   となっているフォルダが 3 つ存在するため、`i`, `j` からフォルダ名を`snap+{i+j}`とした

4. 元データの中身 - 縦 1025 \* 横 513 のデータ - 各種パラメータ 1. density : 密度 1. enstrophy : エンストロフィー 1. magfieldx : 磁場 x 方向 1. magfieldy : 磁場 y 方向 1. magfieldz : 磁場 z 方向 1. pressure : 圧力 1. velocityx : 速度 x 方向 1. velocityy : 速度 y 方向 1. velocityz : 速度 z 方向
   <br>
   <br>

## 2. データの加工

### 2.1. 元データを各種パラメータに分割する

- /src/Processing 配下にある`separator.py` を実行する
- 元データをそれぞれのデータに分割するプログラム
- `BasePath.bat` にてディレクトリの生成を一括で行っている
- 各データファイルの生成処理毎に、生成したファイルを移動している
- `.py`　と `.ipynb` では作業ディレクトリの位置が違うので要注意
- 出力先：`/snaps/snap{i}{j}/*/*`

```cmd
./Magnetic> python ./src/Processing/separator.py
```

<br>
<br>

### 2.2. binary を .npy に変換

- `/src/Processing/snap2npy.py` を実行する
- numpy に変換し、教師データの元にする
- 縦 1025 _ 横 513 のデータを、縦 625 (200~825)(上下 200 を切り取る) _ 横 257 (左右 保存) に加工している

```cmd
./Magnetic> python ./src/Processing/snap2npy.py
```

## 3. 流線の可視化

### 3.1. Heatmap

- 使用メソッド

  - plt.streamplot
  - plt.contour
  - sns.heatmap
  - cv2.cvtColor

  出力先：

  - `/images/visualization/heatmap`
  - `/images/visualization/edges`
  - `/images/visualization/Energy_magfield`
  - `/images/visualization/Energy_vector`

<br>
<br>

### 3.2. AVS

処理ファイル：`/src/Visualization/AVS`
出力先：`/images/AVS/*`

<br>
<br>

### 3.3. StreamLine

処理ファイル：`/src/StreamLines`
出力先： `/images/StreamLines/*`

<br>
<br>

### 3.4. LIC

- LIC 法を用いて可視化する
- 縦 625 \* 横 256 の加工済み npy データを使う
- 一枚 15 分 ~ 20 分程度

  処理ファイル：`/src/Visualization/LIC/LIC.py`
  出力先

  - `/images/LIC/snap{i}/left/*`
  - `/images/LIC/snap{i}/right/*`

<br>
<br>

## 4. 機械学習

### 4.1. 教師データの作成

#### 4.1.1 ビューワの作成

- 画像から反応点の座標を切り出し、教師データを作成する
- そのための Viewer(HTML) を作成する

処理ファイル：`/src/Processing/viewer/createViewer.py`
出力先：`/src/Processing/viewer/template`

```cmd
./Magnetic> python ./src/Processing/snap2npy.py
```

#### 4.1.2 データの切り取り

- 4.1.1 で作成した viewer を基に教師データを作成する
- 8 セット程度行う

処理ファイル：`/src/Processing/viewer/writer.py`
出力先：`/ML/data/LIC_labels/label_snap_{ij}.csv`

#### 4.1.3 データの統合

- 切り取った座標データを基にマスターを作成する
- (param, job) = (1, 10) と (5, 10) の反応点の座標は大体一致している
  -> 各 dataset, label 毎に座標データを纏める

処理ファイル：`/src/Processing/train/make_train.py`
出力先：`/ML/data/LIC_labels/snap_labels.json`

#### 4.1.3 データ生成

- json ファイルの座標を基に npy データを切り取り、教師データを作成する
- 複数の変数を混合した教師データを作成する

処理ファイル：`/src/Processing/train/fusion_npy.py`
出力先：`/ML/data/snap_files`

### 4.2 機械学習

- 分類器
  - k-Means
  - SVM
  - 非線形 SVM (rbf) (ovo, ovr)
  - k-近傍法
  - XGBoost

処理ファイル：

- `/src/MachineLearning/Training.py`
- `/src/MachineLearning/Tuning.py`

出力先：

- `/ML/models`
- `/ML/result`

### 4.7. CNN

処理ファイル：`/src/MachineLearning/CNN.py`
出力先：

- `/ML/models`
- `/ML/result`
  <br>
  <br>

## ディレクトリ構造

Magnetic/

    ├ .git/
    ├ .vscode/ 	VScodeの設定情報の格納先
    ├ .venv/ 仮想環境の格納先
    ├ cmd/ .batファイル等
    ├ data/  元データ
    ├ images/  画像データの出力先
    ├ logs/ ログ
    ├ ML/ 教師データ
    |   ├ data/ 教師データの作成先
    |   ├ models/ 学習モデルの保存先
    |   └ result/ 学習結果の保存先
    ├ snaps/  パラメータ毎に分解したデータ
    |   ├ snap77/
    |   |   ├ density/
    |   |   |     ├ 00/
    |   |   |     ├ 01/
    |   .   .     .
    |   .   .     .
    |   |   |     └ 14/
    |   |   ├ enstrophy/
    |   |   |     ├ 00/
    |   |   |     ├ 01/
    |   .   .     .
    |   .   .     .
    |   |   |     └ 14/
    |   |   ├ magfieldx/
    |   .   .
    |   .   .
    |   |   └ velocityz
    |   ├ half_left/ 元データの左半分の .npyファイル
    |   ├ half_right/ 元データの右半分の .npyファイル
    |   └ all/ 元データ全部の .npyファイル
    |
    ├ src/
    |   ├─config/ 設定ファイル
    |   ├─MachineLearning/ 機械学習系
    |   ├─Processing/　データの前処理系のプログラム
    |   │  ├─separator/
    |   │  ├─train/
    |   │  └─viewer/
    |   │     └─template
    |   └─Visualization/ 可視化用プログラム
    |      ├─AVS/
    |      ├─LIC/
    |      ├─StreamLines/
    |      └─Visualize/
    |
    ├ main.ipynb
    ├ pyproject.toml
    ├ README.md
    ├ requirement.txt
    └.gitignore
