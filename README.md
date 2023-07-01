# 磁気リコネクション
<right>written by Kuniya Ota</right>

23卒の先輩方が行った研究のデータをもとに、リファクタリングを行ったもの

※ \Magnetic を作業ディレクトリとしてください
```
> pwd  
.\Magnetic  

```
<br>

## 1. 元データ
1. `.\etc\mkdir.bat` を実行する  

2. 元データを `.\data` 配下に保存する  
    ```
    \data\20220624.CITM\ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0
    ```
    となっているフォルダが3つ存在するため、`i`, `j` からフォルダ名を`snap+{i+j}`とした

3. 元データの中身
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
1.  元データを各種パラメータに分割する
    - \src\Processing 配下にある`separater.py` を実行する
    - 元データをそれぞれのデータに分割するプログラム
    - `mkdirs.bat` にてディレクトリの生成を一括で行っている
    - 各データファイルの生成処理毎に、生成したファイルを移動している
    - `.py`　と `.ipynb` では作業ディレクトリの位置が違うので要注意

    ```cmd
    .\Magnetic> python .\src\Processing\separater.py

    ```
    - 出力先：`\snaps\snap{i}{j}\*\*`

2. binary を .npy に変換
    - `\src\Processing\snap2npy.py` を実行する
    - numpy に変換し、教師データの元にする
    - それなりに時間がかかる
    ```cmd
    .\Magnetic> python .\src\Processing\snap2npy.py

    ```

<br>
<br>

## 3. 流線の可視化
### 1. Heatmap
処理ファイル：`\src\Visualization`  
```python
from SnapData import SnapData
from Visualization import Visualize
from src.params import IMGOUT, SNAP_PATH

target_path = SNAP_PATH + f"\\snap{i}"
density_files = glob(target_path + f"\\density\\*\\*")
velocityx_files = glob(target_path + f"\\velocityx\\*\\*")
velocityy_files = glob(target_path + f"\\velocityy\\*\\*")
magfieldx_files = glob(target_path + f"\\magfieldx\\*\\*")
magfieldy_files = glob(target_path + f"\\magfieldy\\*\\*")
enstrophy_files = glob(target_path + f"\\enstrophy\\*\\*")


viz = Visualize()    
for target in ["velocityx", "velocityy", "magfieldx", "magfieldy", "density", "enstrophy"]:
    for path in enstrophy_files:
        viz.drawHeatmap(path)
        viz.drawEdge(path)

for dens_path, vx_path, vy_path in zip(density_files, velocityx_files, velocityy_files):
    viz.drawEnergy_for_velocity(dens_path, vx_path, vy_path)

for magx_path, magy_path in zip(magfieldx, magfieldy):
    viz.drawEnergy_for_magfield(magx_path, magy_path)

```

出力先：
- `\imgout\visualization\heatmap`
- `\imgout\visualization\edges`
- `\imgout\visualization\Energy_magfield`
- `\imgout\visualization\Energy_vectory`

<br>
<br>

### 2. AVS
処理ファイル：`\src\AVS`  


出力先：`\imgout\\*`

<br>
<br>

### 3. StreamLine
処理ファイル：`\src\StreamLines`  


出力先：`\imgout\\*`

<br>
<br>

### 4. LIC
処理ファイル：`\src\LIC`  


出力先：`\imgout\\*`

<br>
<br>

## 4. 機械学習
### 1. 教師データの作成
処理ファイル：`\src\Visualization`  


出力先：`\MLdata\\*`

<br>
<br>

### 2. SVM

出力先：`\MLres\\*`

<br>
<br>

### 3. 非線形SVM

出力先：`\MLres\\*`

<br>
<br>

### 4. k-Means

出力先：`\MLres\\*`

<br>
<br>

### 5. XGBoost

出力先：`\MLres\\*`

<br>
<br>

### 6. CNN

出力先：`\MLres\\*`

<br>
<br>

## ディレクトリ構造
Magnetic/  

    ├ .git/  
    ├ data/  元データ  
    ├ imgout/  画像データの出力先  
    ├ MLdata/ 教師データ  
    ├ MLres/ 学習結果の保存先  
    ├ snaps/  パラメータ毎に分解したデータ  
    |   ├ density/  
    |   |     ├ 00/  
    |   |     ├ 01/  
    .   .     .  
    .   .     .  
    |   |     └14/  
    |   ├ enstrophy/  
    |   |     ├ 00/  
    |   |     ├ 01/  
    .   .     .  
    .   .     .  
    |   |     └14/  
    |   ├ magfieldx/  
    .   .  
    .   .  
    |   └velocityz  
    |  
    ├ src/  
    |   ├ AVS/ AVS可視化  
    |   ├ k_means/ k近傍法  
    |   ├ LIC/ LIC可視化  
    |   ├ Processing/ 元データの加工  
    |   ├ SteamLines/ Stream可視化  
    |   └ Visualization/ 元データの可視化  
    |   
    |   .  
    |   .   
    |  
    ├ .env  
    ├ main.ipynb  
    ├ README.md  
    └.gitignore  

```

_visvec.ipynb

avs
-> _imgsplit.ipynb

stream
-> _streamplot.ipynb

LIC
-> ohnolic.py (bynaly -> bmp)

make2data
-> makeviewer2.py, writer.py (bmp -> bmp)
-> _clustering.ipynb (bmp -> csv)(k-means)
-> makesepnpy.ipynb (csv -> npy)
-> fusionnpy.ipynb (fusion npy)

ML
-> MLs.py
-> doML.py
-> XGBoost.ipynb
```
