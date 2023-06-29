# 磁気リコネクション
<right>written by Kuniya Ota</right>

23卒の先輩方が行った研究のデータをもとに、リファクタリングを行ったものをここに残しておく

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
  
出力先：```\snaps\\*```


## 3. データの可視化
### 1. AVS
出力先：`\imgout\\*`


### 2. StreamLine
出力先：`\imgout\\*`


### 3. LIC
出力先：`\imgout\\*`


### 4. 教師データの作成
出力先：`\MLdata\\*`



## 5. 機械学習
### 1. SVM
出力先：`\MLres\\*`


### 2. 非線形SVM
出力先：`\MLres\\*`


### 3. 
出力先：`\MLres\\*`


### 4. XGBoost
出力先：`\MLres\\*`


### 5. CNN
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
