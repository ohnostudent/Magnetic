# 磁気リコネクション
<right>written by Kuniya Ota</right>

23卒の先輩方が行った研究のデータをもとに、リファクタリングを行ったものをここに残しておく

## 1. 元データ
data/ 配下に保存する  
```
data\20220624.CITM\ICh.target=50.ares=1.0d-{i}.adiffArt=1.0d-{j}.h00.g00.BCv1=0.0
```
となっているフォルダが3つ存在するため、`i`, `j` からフォルダ名を`snap+{i+j}`とした


## 2. データの加工
親ディレクトリを作業場とし、src/processing/ 配下にある `separater.py` を実行する  

1. .bat にてディレクトリの生成を一括で行っている
2. 各データの処理毎に、生成したファイルを移動している
3. .py　と .ipynb では作業ディレクトリの位置が違うので要注意


## 3. データの可視化



## 4. 




## ディレクトリ構造
research/
    ├cln/  c言語のツール(著大野先生)
    ├data/  元データ
    ├imgout/  画像,そのほかの出力先
    ├snap/  分解後のデータ
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
    ├src/  ipynbやモジュール
    |   ├mymodule/
    |   |     ├LIC/　きれいに出てくる流線可視化
    |   |     ├SteamLines/ うまくいかない流線可視化/
    |   |     ├MLs.py 機械学習のモジュール。
    |   |     └myfunc.py　諸々の関数。
    |   .   
    |   .   
    |
    ├MLres/ 学習結果の保存先
    ├README.md
    ├.gitignore
    └.git
