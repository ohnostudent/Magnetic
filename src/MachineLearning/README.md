## 4. 機械学習

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

### 4.2. k-Means
    - kMeans

```python
import os
import sys
import numpy as np
from glob import glob

sys.path.append(os.getcwd() + "\src")

from config.params import SNAP_PATH, datasets, variable_parameters
from k_means.KMeans import ClusteringMethod
cluster = ClusteringMethod()

for dataset in datasets:
    for target in variable_parameters:
        path_list = glob(SNAP_PATH + f"/snap{dataset}/{target}/*/*")
        num_of_data = len(path_list)  # リコネクションがない画像の枚数

        temp_data = cluster.compress(cluster.loadSnapData(path_list[0], z=3))
        IMGSHAPE = temp_data.shape

        N_col = IMGSHAPE[0] * IMGSHAPE[1] * 1  # 行列の列数
        X_train = np.zeros((num_of_data, N_col))  # 学習データ格納のためゼロ行列生成
        y_train = np.zeros((num_of_data))  # 学習データに対するラベルを格納するためのゼロ行列生成

        # リコネクションがない画像を行列に読み込む
        for idx, item in enumerate(path_list[:10]):
            X_train[idx, :] = cluster.load_regularize(item)
            y_train[idx] = 0  # リコネクションがないことを表すラベル

        X_train_pca = cluster.PCA(X_train)
        cluster_labels = cluster.KMeans(X_train_pca)
        df_re = cluster.save_result(cluster_labels, path_list, dataset)
        # display(df_re)
```
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
