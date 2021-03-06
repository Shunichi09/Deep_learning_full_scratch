# 標準ライブラリ系
import sys
import numpy as np
import pickle
import os 

# レイヤー
from layers_sin import TimeRNN, TimeAffine, TimeLSTM, TimeIdentifyWithLoss

class BaseModel:
    '''
    基本のネットワーク動作
    '''
    def __init__(self):
        self.params, self.grads = None, None

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_params(self, file_name=None):
        '''
        保存用
        '''
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        params = [np.array(p, dtype='f2') for p in self.params]
        '''
        if GPU:
            params = [to_cpu(p) for p in params]
        '''

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name=None):
        '''
        loadする
        '''
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            file_name = file_name.replace('/', os.sep)

        if not os.path.exists(file_name):
            raise IOError('No file: ' + file_name)

        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        '''
        if GPU:
            params = [to_gpu(p) for p in params]
        '''

        for i, param in enumerate(self.params):
            param[...] = params[i] # loadしたやつをself.paramsに入れている

class SimpleRnn(BaseModel):
    '''
    NN構成：simple RNN ⇒ Affine ⇒　identify with loss
    '''
    def __init__(self, input_size, hidden_size, output_size):
        D, H, O = input_size, hidden_size, output_size # 入力の次元，隠れ層の次元，出力の次元
        rn = np.random.randn

        # 重みの初期化
        rnn_Wx = (rn(D, H) / 10).astype('f')
        rnn_Wh = (rn(H, H) / 10).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, O) / 10).astype('f')
        affine_b = np.zeros(O).astype('f')

        # レイヤの生成
        self.layers = [
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=False), 
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeIdentifyWithLoss()
        self.rnn_layer = self.layers[0]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        # サイズ保存しておく
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts): # 教師，入力ともに三次元
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

class RnnLSTM(BaseModel):
    '''
    NN構成：LSTMs ⇒ Affine ⇒　identify with loss
    '''
    def __init__(self, input_size, hidden_size, output_size):
        D, H, O = input_size, hidden_size, output_size # 入力の次元，隠れ層の次元，出力の次元
        rn = np.random.randn

        # 重みの初期化
        # 基本はこの式
        # x（バッチ×時系列×次元） --> x * Wx(Embedding) -->  hWh + xWx + b = h --> h（バッチ×時系列×次元）* Wx(Affine) --> 出力
        lstm_Wx = (rn(D, 4 * H) / 10).astype('f') # LSTMは複雑そうに見えて重みはこれだけ！後は内部で保存されてる
        lstm_Wh = (rn(H, 4 * H) / 10).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, O) / 10).astype('f')
        affine_b = np.zeros(O).astype('f')

        # レイヤの生成
        self.layers = [
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeIdentifyWithLoss()
        self.lstm_layer = self.layers[0]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        # サイズ保存しておく
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size


    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self): # the reset!!
        self.lstm_layer.reset_state()