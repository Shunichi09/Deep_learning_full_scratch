# 標準ライブラリ系
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import datetime
import math

# NN関係
from figure import Formal_mul_ploter
from NN_temp import SimpleRnn, SimpleRnngen, RnnLSTM, RnnLSTMgen # ネットワーク構成
from optimizer_trainer_temp import SGD, Trainer, RnnlmTrainer # 最適化手法

# dataをreadするクラス 
class Data(): 
    def __init__(self, path):
        self.path = path

    def read(self, data_names):
        '''
        dataの読み込み
        '''
        data = pd.read_csv(self.path, engine='python')
        data = data.values # numpyへ変換
        self.data_names = data_names

        # 各状態を格納
        self.data_dic = {}
        for i, name in enumerate(self.data_names):
            self.data_dic[name] = data[:, i]

        # print(data)

        # 正規化
        self._min_max_normalization()

        # 文字列だけ加工します
        for i in range(len(self.data_dic['date'])):
            self.data_dic['date'][i] = datetime.datetime.strptime(self.data_dic['date'][i], '%Y/%m/%d')
        
        return self.data_dic

    def read_sample_data(self, name):
        '''
        sin波（ノイズ付き）の作成
        '''
        total_size = 1000
        # 各状態を格納
        self.data_dic = {}
        self.data_dic['x'] = [i for i in range(total_size)]
        self.data_dic['end'] = []

        # 何ステップか
        T = 100

        for i in range(total_size):
            noise = 0.05 * np.random.uniform(low=-1.0, high=1.0)
            self.data_dic['end'].append(math.sin((i/T) * 2 * math.pi) + noise)

        # 正規化
        # self._min_max_normalization()
        
        return self.data_dic

    def _min_max_normalization(self):
        '''
        dataの最大最小正規化を行う
        '''
        for name in self.data_names:
            if name ==  'date':
                continue

            MAX = np.max(self.data_dic[name])
            MIN = np.min(self.data_dic[name])

            print('name = {0} | MAX = {1} | MIN = {2} '.format(name, MAX, MIN))

            for k in range(len(self.data_dic[name])):
                self.data_dic[name][k] = (self.data_dic[name][k] - MIN) / (MAX - MIN)

    def make_data_set_for_NN(self, T, name):
        '''
        datasetを作成するクラス，保持しないタイプ
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

        # print(x_train, t_train, x_test, t_test)

        return x_train, t_train, x_test, t_test
    
    def make_data_set_for_RNN(self, name):
        '''
        datasetを作成するクラス，保持するタイプ
         name = dataの名前
        '''
        rate = 0.9  # datasetの割合
        data_size = len(self.data_dic[name]) 
        train_size = int(data_size * rate)

        x_train = self.data_dic[name][:train_size-1]
        t_train = self.data_dic[name][1:train_size]

        x_train = np.array(x_train, dtype='f')
        t_train = np.array(t_train, dtype='f')

        x_test = []
        t_test = []

        # Test_data
        x_test = self.data_dic[name][train_size:-1]
        t_test = self.data_dic[name][train_size+1:]

        x_test = np.array(x_test, dtype='f')
        t_test = np.array(t_test, dtype='f')

        # print(x_train, t_train, x_test, t_test)

        print('data_size = {0} | train_size = {1} '.format(data_size, train_size))

        return x_train, t_train, x_test, t_test

def main():

    names = ['date', 'temp']
    
    # dataの読み込み
    # LINE
    path = 'data.csv'
    tempre_data_editer = Data(path)
    tempre_data_editer.read(names)
    
    # データの可視化
    for key in [tempre_data_editer]:
        x = key.data_dic[names[0]]
        y = [key.data_dic[names[1]]]
        x_label_name = 'date'
        y_label_name = 'temp'
        y_names = names[1]
        ploter = Formal_mul_ploter(x, y, x_label_name, y_label_name, y_names)
        ploter.mul_plot()

    # ハイパーパラメータの設定
    batch_size = 20 # バッチサイズ
    input_size = 1 # 入力の次元
    hidden_size = 100 # 隠れ層の大きさ
    output_size = 1 # 出力の次元
    time_size = 25 # Truncated BPTTの展開する時間サイズ，RNNのステップ数 # 20
    lr = 0.01 # 学習率 0.01
    max_epoch = 15000 # 最大epoch

    # dataset作成
    x_train, t_train, x_test, t_test = tempre_data_editer.make_data_set_for_RNN('temp')

    # モデルの生成
    # simpleRnnの場合
    model = SimpleRnn(input_size, hidden_size, output_size)
    predicter = SimpleRnngen(input_size, hidden_size, output_size)
    # LSTMの場合
    # model = RnnLSTM(input_size, hidden_size, output_size)
    # predicter = RnnLSTMgen(input_size, hidden_size, output_size)

    # 最適化
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)
    
    # 勾配クリッピングを適用して学習
    trainer.fit(x_train, t_train, max_epoch, batch_size, time_size)
    trainer.plot()

    # パラメータの保存
    model.save_params()
    
    # ネットワークを用いて次のデータを予測
    predicter.load_params('SimpleRnn.pkl')
    input_x = np.array(x_test[:time_size].reshape(1, time_size, input_size), dtype='f') # 始めの入力を作成

    predict_y, ans_t = predicter.generate(input_x, t_test[time_size-1:], sample_size=len(t_test[time_size-1:]))

    plt.plot(range(len(x_train[-50:])), x_train[-50:], label='data', marker='.')
    plt.plot(range(time_size + len(x_train[-50:]), time_size + len(x_train[-50:]) + len(predict_y)), predict_y, label='pre', marker='.')
    plt.plot(range(len(x_train[-50:]),  len(x_train[-50:]) + len(x_test)), x_test, label='ans', marker='.')

    plt.show()

if __name__ == '__main__':
    main()