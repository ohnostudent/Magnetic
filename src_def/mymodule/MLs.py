import numpy as np 
import os
import random 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import glob 
from mymodule import myfunc as mf
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC 
import xgboost as xgb 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
import re

#selfついてるのとついてないのとでバグあるかも

class ML:
    def __init__(self, TARGET, ALTIMAGES0, ALTIMAGES1, SOURCE0, SOURCE1, IMGSHAPE=(100,10), DO_PCA = False, dilute = False, randomstate = None) -> None:
        self.TARGET = TARGET#magfieldx, pressure,,,
        self.ALTIMAGES0 = ALTIMAGES0
        self.ALTIMAGES1 = ALTIMAGES1
        self.SOURCE0 = SOURCE0#
        self.SOURCE1 = SOURCE1#
        self.JUDGE_COLUMN = "is_reconnecting"#(0,1)
        self.IMGSHAPE = IMGSHAPE#出来れば画像サイズはすべて同じで合ってほしい。違うサイズが混じる場合は最も多いサイズを指定すること
        self.DO_PCA = DO_PCA
        self.randomstate = randomstate
        # def compress(array, LEVEL=1):
        #     return mf.convolute(array,mf.ave_carnel(LEVEL), stride = LEVEL)
        # temp = compress(mf.load())
        # IMGSHAPE = temp.shape



        """
        labelcsvs0 = glob.glob(LABEL_SOURCE0+"*")############
        labelcsvs1 = glob.glob(LABEL_SOURCE1+"*")############

        columns = ["snappath","dataset","para","job","centerx","centery","xlow","xup","ylow","yup"]###############

        df = pd.DataFrame(columns=columns+[self.JUDGE_COLUMN])
        for labelcsv in labelcsvs0:
            # dftmp = pd.read_csv(labelcsv, index_col=0)[columns]
            dftmp = pd.read_csv(labelcsv)[columns]
            #judgecolum=1 or 0 wo insert
            dftmp[self.JUDGE_COLUMN] = 0
            df = pd.concat([df,dftmp])
        for labelcsv in labelcsvs1:
            # dftmp = pd.read_csv(labelcsv, index_col=0)[columns]
            dftmp = pd.read_csv(labelcsv)[columns]
            #judgecolum=1 or 0 wo insert
            dftmp[self.JUDGE_COLUMN] = 1
            df = pd.concat([df,dftmp])
        """
        #./magfieldx_49.52.6_47.773.npy
        #############
        self.PATH0 = []
        for i in self.SOURCE0:
            self.PATH0.extend(glob.glob(i + self.TARGET +"/*"))
        self.PATH1 = []
        for i in self.SOURCE1:
            self.PATH1.extend(glob.glob(i + self.TARGET +"/*"))
        #############

        # self.PATH0 = (df[df[self.JUDGE_COLUMN] == 0][["dataset","para","job","centerx","centery"]])
        # # self.PATH0["path"] = self.ALTIMAGES0+TARGET+"/_"+self.PATH0['dataset'].astype(str)+"."+ self.PATH0['para'].astype(str)+"."+self.PATH0['job'].astype(str)+"_"+self.PATH0['centerx'].astype(str) +"."+self.PATH0['centery'].astype(str)+".npy"
        # self.PATH0 = list(self.ALTIMAGES0+TARGET+"/"+ TARGET +"_"+self.PATH0['dataset'].astype(str)+"."+ self.PATH0['para'].astype(str)+"."+self.PATH0['job'].astype(str)+"_"+self.PATH0['centerx'].astype(str) +"."+self.PATH0['centery'].astype(str)+".npy")
        # self.PATH1 = (df[df[self.JUDGE_COLUMN] == 1][["dataset","para","job","centerx","centery"]])
        # # self.PATH1["path"] = self.ALTIMAGES1+TARGET+"/_"+self.PATH1['dataset'].astype(str)+"."+ self.PATH1['para'].astype(str)+"."+self.PATH1['job'].astype(str)+"_"+self.PATH1['centerx'].astype(str) +"."+self.PATH1['centery'].astype(str)+".npy"            
        # self.PATH1 = list(self.ALTIMAGES1+TARGET+"/"+ TARGET +"_"+self.PATH1['dataset'].astype(str)+"."+ self.PATH1['para'].astype(str)+"."+self.PATH1['job'].astype(str)+"_"+self.PATH1['centerx'].astype(str) +"."+self.PATH1['centery'].astype(str)+".npy")            

        self.split_testtrain()
        if dilute:
            self.dilute()
        self.load_data()
        self.exePCA()
    def size_dir(self, path):
        return len(glob.glob(path))
    def split_testtrain(self):#############別の方法も実装するかも
        def split4977(paths):
            array49 = []
            array77 = []
            for path in paths:
                if self.TARGET + "_49" in path:
                    array49.append(path)
                elif self.TARGET + "_77" in path:
                    array77.append(path)
                else:
                    print("split4977 error")
            return array49, array77
            # return np.array(array49), np.array(array77)
        
        self.PATH0TRAIN, self.PATH0TEST = split4977(self.PATH0)
        self.PATH1TRAIN, self.PATH1TEST = split4977(self.PATH1)
        # self.PATH0TRAIN, self.PATH0TEST = train_test_split(self.PATH0, test_size=0.3, shuffle=True)#, stratify=self.PATH0)
        # self.PATH1TRAIN, self.PATH1TEST = train_test_split(self.PATH1, test_size=0.3, shuffle=True)#, stratify=self.PATH1)
    def dilute(self):
        #ランダムな領域の切り取りを行う関数
        def random_crop(imagearay, size=0.8):
            height, width, _ = imagearay.shape
            crop_size = int(min(height, width) * size)

            top = np.random.randint(0, height - crop_size)
            left = np.random.randint(0, width - crop_size)
            bottom = top + crop_size
            right = left + crop_size
            imagearay = imagearay[top:bottom, left:right,:]
            #motonosaizunimoosu
            return imagearay
        def altarray_save(item, temp_output_dir):
            img = mf.load(item, z=3)
            file_name = os.path.basename(item)
            img_flip = np.flipud(img) # 画像の上下反転
            np.save(temp_output_dir + "flip_" + file_name , img_flip) # 画像保存
            img_mirror = np.fliplr(img) # 画像の左右反転
            np.save(temp_output_dir + "mirr_" + file_name , img_mirror) # 画像保存
            # img_T = mf.resize(img.T, self.IMGSHAPE) # 画像の上下左右反転
            # print(img.shape,img.T.shape, img_T.shape, self.IMGSHAPE)
            # np.save(temp_output_dir + "trns_" + file_name , img_T) # 画像保存
            # img_crop = random_crop(img) # 画像の切り取り
            # img_crop = img_crop.resize((256, 256)) # 元のサイズに戻す
            # img_crop.save(temp_output_dir + file_name + "_crop.png") # 画像保存
        
        #リコネクションがない画像ファイルのパスのリストを取得
        files = self.PATH0TRAIN
        #出力ディレクトリのパス
        if not os.path.exists(self.ALTIMAGES0):
            raise "ALTIMGES0 is not correct"
        temp_output_dir = self.ALTIMAGES0+self.TARGET+"/"
        if  not os.path.exists(temp_output_dir):
                os.mkdir(temp_output_dir)
        for item in files:
            altarray_save(item,temp_output_dir)
        #リコネクションがある画像ファイルのパスのリストを取得
        files = self.PATH1TRAIN
        #出力ディレクトリのパス
        if not os.path.exists(self.ALTIMAGES1):
            raise "ALTIMGES1 is not correct"
        temp_output_dir = self.ALTIMAGES1+self.TARGET+"/"
        if  not os.path.exists(temp_output_dir):
            os.mkdir(temp_output_dir)
        for item in files:
            altarray_save(item,temp_output_dir)
    def load_data(self):
        # 訓練データ
        #snap49のdilute分だけ訓練データにする
        self.ALLTARINDATA0 = list(set(list(glob.glob(self.ALTIMAGES0+self.TARGET+f"/*{self.TARGET}_49*")) + self.PATH0TRAIN))
        self.ALLTARINDATA1 = list(set(list(glob.glob(self.ALTIMAGES1+self.TARGET+f"/*{self.TARGET}_49*")) + self.PATH1TRAIN))
        # self.ALLTARINDATA0 = list(set(list(glob.glob(self.ALTIMAGES0+self.TARGET+"/*")) + self.PATH0TRAIN))
        # self.ALLTARINDATA1 = list(set(list(glob.glob(self.ALTIMAGES1+self.TARGET+"/*")) + self.PATH1TRAIN))
        num_of_data_clear = len(self.ALLTARINDATA0) # リコネクションがない画像の枚数
        num_of_data_cloudy = len(self.ALLTARINDATA1) # リコネクションがある画像の枚数
        num_of_data_total = num_of_data_clear + num_of_data_cloudy # 学習データの全枚数

        N_col = self.IMGSHAPE[1]*self.IMGSHAPE[0]*1 # 行列の列数
        self.X_train = np.zeros((num_of_data_total, N_col)) # 学習データ格納のためゼロ行列生成
        self.y_train = np.zeros((num_of_data_total)) # 学習データに対するラベルを格納するためのゼロ行列生成
        self.path_train = list("" for i in range(num_of_data_total)) # 学習データに対するpathを格納するためのゼロ行列生成

        # リコネクションがない画像を行列に読み込む
        path_list = self.ALLTARINDATA0
        i_count = 0

        def load_regularize(item):
            type = item[-4:]
            # print(item)
            if type == ".npy":
                im = np.load(item)
            elif type == ".npz":
                print("npz doesnot supported")
                return
            elif type == ".jpg":
                im = Image.open(item).convert('L')
                im =im.resize(self.IMGSHAPE) # 画像のサイズ変更
                im = np.ravel(np.array(im)) # 画像を配列に変換
                # im = im_array/255. # 正規化 
            else:
                im = mf.load(item, z=3)
            # img_resize = compress(im)
            img_resize = mf.resize(im, self.IMGSHAPE)
            if im.shape != self.IMGSHAPE:
                print("resized:",item,im.shape,"to" ,self.IMGSHAPE)
            return ((img_resize - min(img_resize.flat)) / max(img_resize.flat)).flat # 正規化

        for item in path_list:
            self.X_train[i_count,:] = load_regularize(item)
            self.y_train[i_count] = 0 # リコネクションがないことを表すラベル
            self.path_train[i_count] = item
            i_count += 1

        # リコネクションがある画像を行列に読み込む
        path_list = self.ALLTARINDATA1

        for item in path_list:
            self.X_train[i_count,:] = load_regularize(item)
            self.y_train[i_count] = 1 # リコネクションがあることを表すラベル
            self.path_train[i_count] = item
            i_count += 1
        

        # テストデータ
        num_of_data_clear = len(self.PATH0TEST) # リコネクションがない画像の枚数
        num_of_data_cloudy = len(self.PATH1TEST) # リコネクションがある画像の枚数
        num_of_data_total = num_of_data_clear + num_of_data_cloudy # テストデータの全枚数
        
        N_col = self.IMGSHAPE[1]*self.IMGSHAPE[0]*1 # 行列の列数(RGBなら*3)
        self.X_test = np.zeros((num_of_data_total, N_col)) # テストデータ格納のためゼロ行列生成
        self.y_test = np.zeros(num_of_data_total) # テストデータに対するラベルを格納するためのゼロ行列生成
        self.path_test = list("" for i in range(num_of_data_total)) # テストデータに対するpathを格納するためのゼロ行列生成
        
        # リコネクションがない画像を行列に読み込む
        path_list = self.PATH0TEST 
        i_count = 0
        for item in path_list:
        
            self.X_test[i_count,:] = load_regularize(item)
            self.y_test[i_count] = 0 # リコネクションがないことを表すラベル
            self.path_test[i_count] = item
            i_count += 1
        
        # リコネクションがある画像を行列に読み込む
        path_list = self.PATH1TEST
        
        for item in path_list:
            self.X_test[i_count,:] = load_regularize(item)
            self.y_test[i_count] = 1 # リコネクションがあることを表すラベル
            self.path_test[i_count] = item
            i_count += 1 
    def exePCA(self):
        if self.DO_PCA:
            N_dim =  100 #100列に落とし込む

            pca = PCA(n_components=N_dim, random_state=self.randomstate)

            self.X_train_pca = pca.fit_transform(self.X_train)
            self.X_test_pca = pca.transform(self.X_test)

            print('PCA累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
        else:
            self.X_train_pca = self.X_train
            self.X_test_pca = self.X_test
            # print("PCA 非実施")
    
##################################
#以下、学習の関数。手動で実行すること。

    def modelreturns(self,model):
        pred = model.predict(self.X_test)
        mlres = pd.DataFrame(np.array([self.path_test, self.y_test, pred]).T, columns=["path", "y", "predict"])
        report = classification_report(self.y_test, pred)
        return model, mlres, report
    def printscore(self, modelands):
        print("Train :", modelands[0].score(self.X_train_pca,  self.y_train))
        print("Test :", modelands[0].score(self.X_test_pca, self.y_test))
        print(modelands[2])

    def linearSVC(self):
        model = LinearSVC(C=0.3, random_state=self.randomstate) # インスタンスを生成
        model.fit(self.X_train_pca, self.y_train) # モデルの学習
        return self.modelreturns(model)
    def kneighbors(self):
        n_neighbors = int(np.sqrt(6000))  # kの設定
        model = KNeighborsClassifier(n_neighbors = n_neighbors)  
        model.fit(self.X_train_pca, self.y_train) # モデルの学習
        return self.modelreturns(model)
    def rbfSVC(self):
        model = SVC(C=0.3, kernel='rbf', random_state=self.randomstate) # インスタンスを生成 
        model.fit(self.X_train_pca, self.y_train) # モデルの学習
        return self.modelreturns(model)
    def XGBoost(self):
        model = xgb.XGBClassifier(n_estimators=80, max_depth=4, gamma=3) # インスタンスの生成
        model.fit(self.X_train_pca, self.y_train) # モデルの学習
        return self.modelreturns(model)



    def testmodel(self):
        n_neighbors = int(np.sqrt(6000))  # kの設定
        model = KNeighborsClassifier(n_neighbors = n_neighbors)  
        model.fit(self.X_train_pca, self.y_train) # モデルの学習
        return self.modelreturns(model)