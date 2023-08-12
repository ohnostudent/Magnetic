# 磁気リコネクション
<right>written by Kuniya Ota</right>

23卒の先輩方が行った研究のデータをもとに、リファクタリングを行ったもの

※ /Magnetic を作業ディレクトリとしてください
```
> pwd
./Magnetic

```
※ 場合によっては以下のコードを追記してください
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

2. `/bin/mkdir.bat` を実行する

3. 元データを `./data` 配下に保存する
    ```
    /data/ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0
    ```
    となっているフォルダが3つ存在するため、`i`, `j` からフォルダ名を`snap+{i+j}`とした

4. 元データの中身
    - 縦1025 * 横513 のデータ
    - 各種パラメータ
        1. density : 密度
        1. enstrophy : エンストロフィー
        1. magfieldx : 磁場x方向
        1. magfieldy : 磁場y方向
        1. magfieldz : 磁場z方向
        1. pressure : 圧力
        1. velocityx : 速度x方向
        1. velocityy : 速度y方向
        1. velocityz : 速度z方向
<br>
<br>

## 2. データの加工
### 2.1. 元データを各種パラメータに分割する
    - /src/Processing 配下にある`separator.py` を実行する
    - 元データをそれぞれのデータに分割するプログラム
    - `mkdirs.bat` にてディレクトリの生成を一括で行っている
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
    - それなりに時間がかかる
    - 縦1025 * 横513 のデータを、縦625 (200~825)(上下200を切り取る) * 横257 (左右 保存) に加工している
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
処理ファイル：`/src/AVS`
出力先
- `/images/AVS/*`

<br>
<br>

### 3.3. StreamLine
処理ファイル：`/src/StreamLines`
出力先
- `/images/StreamLines/*`

<br>
<br>

### 3.4. LIC
- LIC法にて可視化する
- 縦625 * 横256 の加工済みnpyデータを使う
- snaps/half_left/snap77/magfield に保存したデータをすべて処理するために、3.9GHz, 10並列で15時間程度かかる

    処理ファイル：`/src/LIC`
    出力先
    - `/images/LIC/snap{i}/left/*`
    - `/images/LIC/snap{i}/right/*`

<br>
<br>

## 4. 機械学習
### 4.1. 教師データの作成

#### 4.1.1 ビューワの作成
処理ファイル：`/src/Processing/viewer/createViewer.py`
出力先
- `/MLdata/*`
```cmd
./Magnetic> python ./src/Processing/snap2npy.py

```

#### 4.1.2 データの切り取り
1. `/src/Processing/viewer/writer.py` を実行


#### 4.1.3 データの合成
- 画像


### 4.2. k-Means
    - kMeans

<br>
<br>


### 4.3. SVM

出力先
- `/MLres/*`

<br>
<br>

### 4.4. 非線形SVM

出力先
- `/MLres/*`

<br>
<br>

### 4.5. k-近傍法

出力先
- `/MLres/*`

<br>
<br>

### 4.6. XGBoost

出力先
- `/MLres/*`

<br>
<br>

### 4.7. CNN

出力先
- `/MLres/*`

<br>
<br>

## ディレクトリ構造
Magnetic/

    ├ .git/
    ├ .vscode/ 	VScodeの設定情報の格納先
    ├ .venv/ 仮想環境の格納先
    ├ bin/ .batファイル等
    ├ config/ 各種設定ファイルを格納
    ├ data/  元データ
    ├ images/  画像データの出力先
    ├ logs/ ログ
    ├ ML/ 教師データ
    |   ├ data/ 教師データの作成先
    |   └ result/ 学習結果の保存先
    ├ snaps/  パラメータ毎に分解したデータ
    |   ├ snap77/
    |   |   ├ density/
    |   |   |     ├ 00/
    |   |   |     ├ 01/
    |   .   .     .
    |   .   .     .
    |   |   |     └14/
    |   |   ├ enstrophy/
    |   |   |     ├ 00/
    |   |   |     ├ 01/
    |   .   .     .
    |   .   .     .
    |   |   |     └14/
    |   |   ├ magfieldx/
    |   .   .
    |   .   .
    |   |   └ velocityz
    |   ├ half_left/ 元データの左半分の .npyファイル
    |   ├ half_right/ 元データの右半分の .npyファイル
    |   └ all/ 元データ全部の .npyファイル
    |
    ├ src/
    |   ├ AVS/ AVS可視化
    |   ├ k_means/ k近傍法
    |   ├ LIC/ LIC可視化
    |   |   ├ libs/ 各種モジュールの格納先
    |   |   └ temp/ .tempファイルの作成先
    |   ├ Processing/ 元データの加工
    |   |   ├ cln/ c言語系モジュールの格納先
    |   |   ├ libs/ 各種モジュールの格納先
    |   |   └ viewer/ viewer作成に関するモジュールの格納先
    |   ├ SteamLines/ Stream可視化
    |   |   └ libs/ 各種モジュールの格納先
    |   └ Visualization/ 元データの可視化
    |
    |   .
    |   .
    |
    ├ .env
    ├ main.ipynb
    ├ README.md
    ├ requirement.txt
    └.gitignore


```

ML
-> MLs.py
-> doML.py
-> XGtune.ipynb
```
