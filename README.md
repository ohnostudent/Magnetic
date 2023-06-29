# 磁気リコネクション
<right>written by Kuniya Ota</right>

23卒の先輩方が行った研究のデータをもとに、リファクタリングを行ったもの


## 1. 元データ
data/ 配下に保存する  
```
\data\20220624.CITM\ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0
```
となっているフォルダが3つ存在するため、`i`, `j` からフォルダ名を`snap+{i+j}`とした


## 2. データの加工
親ディレクトリを作業場とし、\src\processing\ 配下にある `separater.py` を実行する  

1. .bat にてディレクトリの生成を一括で行っている
2. 各データの処理毎に、生成したファイルを移動している
3. .py　と .ipynb では作業ディレクトリの位置が違うので要注意

```
> pwd  
.\GraduationResearch\Magnetic  

> python .\src\Processing\separater.py

```

出力先：```\snaps\\*```


## 3. 流線の可視化
### 1. Heatmap
処理ファイル：`\src\Visualization`  
```python
from SnapData import SnapData
from src.params import IMGOUT, SNAP_PATH

target_path = SNAP_PATH + f"\\snap{i}"
density_files = glob(target_path + f"\\density\\*\\*")
velocityx_files = glob(target_path + f"\\velocityx\\*\\*")
velocityy_files = glob(target_path + f"\\velocityy\\*\\*")
magfieldx_files = glob(target_path + f"\\magfieldx\\*\\*")
magfieldy_files = glob(target_path + f"\\magfieldy\\*\\*")
enstrophy_files = glob(target_path + f"\\enstrophy\\*\\*")


viz = Visualize()
for dens_path, vx_path, vy_path in zip(density_files, velocityx_files, velocityy_files):
    viz.drawEnergy_for_velocity(dens_path, vx_path, vy_path)

for magx_path, magy_path in zip(magfieldx, magfieldy):
    viz.drawEnergy_for_magfield(magx_path, magy_path)
    
for target in ["velocityx", "velocityy", "magfieldx", "magfieldy", "density", "enstrophy"]:
    for path in enstrophy_files:
        viz.drawHeatmap(path)
        viz.drawEdge(path)

```

出力先：`\imgout\\*`


### 2. AVS
処理ファイル：`\src\AVS`  


出力先：`\imgout\\*`


### 3. StreamLine
処理ファイル：`\src\StreamLines`  


出力先：`\imgout\\*`


### 4. LIC
処理ファイル：`\src\LIC`  


出力先：`\imgout\\*`


## 5. 機械学習
### 1. 教師データの作成
処理ファイル：`\src\Visualization`  


出力先：`\MLdata\\*`

### 2. SVM

出力先：`\MLres\\*`


### 3. 非線形SVM

出力先：`\MLres\\*`


### 4. 

出力先：`\MLres\\*`


### 5. XGBoost

出力先：`\MLres\\*`


### 6. CNN

出力先：`\MLres\\*`



## ディレクトリ構造
research/  

    ├.git/  
    ├data/  元データ  
    ├imgout/  画像データの出力先  
    ├MLres/ 学習結果の保存先  
    ├snaps/  分解後のデータ  
    |   ├density/  
    |   |     ├00/  
    |   |     ├01/  
    .   .     .  
    .   .     .  
    |   |     └14/  
    |   ├enstrophy/  
    |   |     ├00/  
    |   |     ├01/  
    .   .     .  
    .   .     .  
    |   |     └14/  
    |   ├magfieldx/  
    .   .  
    .   .  
    |   └velocityz  
    |  
    ├src/  
    |   ├AVS/ AVS可視化  
    |   ├k_means/ k近傍法  
    |   ├LIC/ LIC可視化  
    |   ├Processing/ 元データの加工  
    |   ├SteamLines/ Stream可視化  
    |   ├Visualization/ 元データの可視化  
    |   └main.ipynb  
    |   .  
    |   .   
    |  
    ├.env  
    ├README.md  
    └.gitignore  

```

_visvec.ipynb

avs
-> _imgsplit.ipynb

stream
-> _streamploy.ipynb

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
