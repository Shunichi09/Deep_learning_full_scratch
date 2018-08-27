# -*- coding: utf-8 -*-
# 標準ライブラリ系
import sys
sys.path.append('..')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

# 関数等その他もろもろ
from functions_shogi import cross_entropy_error, softmax
from layers_shogi import TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss, TimeLSTM, TimeDropout

class BaseModel:
    '''
    基本のネットワーク用
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

class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 重みの初期化
        # 基本はこの式
        # x（バッチ×時系列×次元） --> x * Wx(Embedding) -->  hWh + xWx + b = h --> h（バッチ×時系列×次元）* Wx(Affine) --> 出力
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f') # LSTMは複雑そうに見えて重みはこれだけ！後は内部で保存されてる
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # レイヤの生成
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # すべての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

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


class BetterRnnlm(BaseModel):
    '''
     LSTMレイヤを2層利用し、各層にDropoutを使うモデル
    '''
    def __init__(self, vocab_size=10000, wordvec_size=650,
                 hidden_size=650, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)  # weight tying!!
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg

        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()


# 予想する用のクラス
class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            # 確率と候補をを抽出する
            top = 5 # 上位5つの手まで
            candidate_logit = []
            candidate_move_ids = []
            count = 0

            for i in (-1 * p).argsort(): # indexが返ってくる

                candidate_logit.append(p[i])
                candidate_move_ids.append(i)

                count += 1
                if count >= top:
                    return

            # sampled = np.random.choice(len(p), size=1, p=p)

            # if (skip_ids is None) or (sampled not in skip_ids):
            #     x = sampled
            #     word_ids.append(int(x))

        return candidate_move_ids, candidate_logit

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        x = start_id

        # while len(word_ids) < sample_size:
        x = np.array(x).reshape(1, 1)
        score = self.predict(x).flatten()
        p = softmax(score).flatten()

        # print(p)

        # 確率と候補をを抽出する
        top = 5 # 上位5つの手まで
        candidate_logit = []
        candidate_move_ids = []
        count = 0

        print((-1 * p).argsort())

        for i in (-1 * p).argsort(): # indexが返ってくる
            candidate_logit.append(p[i])
            candidate_move_ids.append(i)

            count += 1
            if count >= top:
                break

            # sampled = np.random.choice(len(p), size=1, p=p)

            # if (skip_ids is None) or (sampled not in skip_ids):
            #     x = sampled
            #     word_ids.append(int(x))
        

        return candidate_move_ids, candidate_logit

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)