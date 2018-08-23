# 標準ライブラリ系
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime
import math

# NN関係
from figure import Formal_mul_ploter
from NN_sin import SimpleRnn, RnnLSTM # ネットワーク構成
from optimizer_trainer_sin import SGD, Trainer # 最適化手法

# dataをreadするクラス 
class Data(): 
    def __init__(self, path):
        self.path = path

    def read(self, data_names):
        '''
        dataの読み込み
        '''
        data = pd.read_csv(self.path, header=None, engine='python')
        data = data.values # numpyへ変換
        self.data_names = data_names

        # 各状態を格納
        self.data_dic = {}
        for i, name in enumerate(self.data_names):
            self.data_dic[name] = data[:, i]

        # 正規化
        self._min_max_normalization()

        # 文字列だけ加工します
        for i in range(len(self.data_dic['date'])):
            self.data_dic['date'][i] = datetime.datetime.strptime(self.data_dic['date'][i], '%Y/%m/%d')
        
        return self.data_dic

    def read_sample_data(self):
        '''
        sin波（ノイズ付き）の作成
        '''
        total_size = 200
        # 各状態を格納
        self.data_dic = {}
        self.data_dic['x'] = [i for i in range(total_size)]
        self.data_dic['y'] = []

        # 何ステップか
        T = 100

        for i in range(total_size):
            noise = 0.05 * np.random.uniform(low=-1.0, high=1.0)
            self.data_dic['y'].append(math.sin((i/T) * 2 * math.pi) + noise)

        # 正規化
        # self._min_max_normalization()
        
        return self.data_dic

    def _min_max_normalization(self):
        '''
        dataの最大最小正規化を行う
        '''
        for name in self.data_names:
            MAX = np.max(self.data_dic[name])
            MIN = np.min(self.data_dic[name])

            print('MAX = {0}'.format(MAX))
            print('MIN = {0}'.format(MIN))

            for k in range(len(self.data_dic[name])):
                self.data_dic[name][k] = (self.data_dic[name][k] - MIN) / (MAX - MIN)

    def make_data_set(self, T, name):
        '''
        datasetを作成するクラス
        引数はT = time_datasize !! name = dataの名前
        '''
        rate = 0.8  # datasetの割合
        data_size = len(self.data_dic[name]) 
        train_size = int(data_size * rate)

        x_train = [] 
        t_train = []

        # Training_data
        for i in range(0, data_size - T - 1):
            x_train.append(self.data_dic[name][i:i+T])
            t_train.append(self.data_dic[name][i+T])

        x_train = np.array(x_train, dtype='f')
        t_train = np.array(t_train, dtype='f')

        x_test = []
        t_test = []

        # Test_data（今は同じにしている）
        for i in range(0, data_size - T - 1):
            x_test.append(self.data_dic[name][i:i+T])
            t_test.append(self.data_dic[name][i+T])
        
        x_test = np.array(x_test, dtype='f')
        t_test = np.array(t_test, dtype='f')

        return x_train, t_train, x_test, t_test

def main():
    # dataの読み込み
    path = 'data.csv'
    data_editer = Data(path)

    # sin波
    data_dic = data_editer.read_sample_data()

    # データの可視化
    x = data_dic['x']
    y = [data_dic['y']]
    x_label_name = 'x'
    y_label_name = 'y'
    y_names = ['y']
    ploter = Formal_mul_ploter(x, y, x_label_name, y_label_name, y_names)
    ploter.mul_plot()

    # ハイパーパラメータの設定
    batch_size = 10 # バッチサイズ
    input_size = 1 # 入力の次元
    hidden_size = 20 # 隠れ層の大きさ
    output_size = 1 # 出力の次元
    time_size = 25 # Truncated BPTTの展開する時間サイズ，RNNのステップ数 # 20
    lr = 0.01 # 学習率 0.01
    max_epoch = 5000 # 最大epoch

    # dataset作成
    x_train, t_train, x_test, t_test = data_editer.make_data_set(time_size, 'y')

    # モデルの生成
    # simpleRnnの場合
    # model = SimpleRnn(input_size, hidden_size, output_size)
    # LSTMの場合
    model = RnnLSTM(input_size, hidden_size, output_size)

    # 最適化
    optimizer = SGD(lr)
    trainer = Trainer(model, optimizer)
    
    # 勾配クリッピングを適用して学習
    trainer.fit(x_train, t_train, time_size, max_epoch, batch_size)
    trainer.plot()

    # パラメータの保存
    model.save_params()
    
    # ネットワークを用いて次のデータを予測
    model.reset_state()
    input_x = np.array(x_test[0, :].reshape(1, time_size, input_size), dtype='f') # 始めの入力を作成
    predict_y = []
    ans_t = []
    for i in range(len(t_test)):
        next_x = model.predict(input_x) # 次のものを予測
        # print(next_x)
        # リスト化
        next_x = list(next_x.flatten())
        input_x = list(input_x.flatten())
        # 要素を削除して追加
        input_x.pop(0)
        input_x.append(next_x[-1])
        # print(t_test[time_size + i], next_x[-1])
        # a = input()
        predict_y.append(next_x[-1])
        ans_t.append(t_test[i])
        
        input_x = np.array(input_x).reshape(1, time_size, input_size)

    plt.plot(range(len(t_test)), predict_y, label='pre')
    plt.plot(range(len(t_test)), ans_t, label='ans')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()