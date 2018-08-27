# -*- coding: utf-8 -*-
# NNのlayers
import numpy as np
import matplotlib.pyplot as plt
from functions_shogi import cross_entropy_error, softmax, sigmoid
import time
import sys

class LSTM:
    def __init__(self, Wx, Wh, b):
        '''
        Wx: 入力`x`用の重みパラーメタ（4つ分の重みをまとめる）
        Wh: 隠れ状態`h`用の重みパラメータ（4つ分の重みをまとめる）
        b: バイアス（4つ分のバイアスをまとめる）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params # パラメータ抽出
        N, H = h_prev.shape # 隠れ状態のサイズ

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b # 内部状態を算出

        f = A[:, :H] # それぞれを挿入する，3列目
        g = A[:, H:2*H] # 2列目
        i = A[:, 2*H:3*H] # 3列目
        o = A[:, 3*H:] # 4列目

        f = sigmoid(f) # forget gate
        g = np.tanh(g) # memorizeする情報
        i = sigmoid(i) # input gate
        o = sigmoid(o) # output gate

        c_next = f * c_prev + g * i # 出力を計算
        h_next = o * np.tanh(c_next) # 次の状態を保持

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache # パラメータ取り出し

        tanh_c_next = np.tanh(c_next) # p246の右端のところ，tanhが必要，掛け算

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2) # 右端のところがちょっと難しいけど，追っていけばできる

        dc_prev = ds * f # 掛け算ノードの逆伝播

        di = ds * g # 同じ
        df = ds * c_prev #  追っていけばできる
        do = dh_next * tanh_c_next # p246の右端のところ，tanhが必要，掛け算
        dg = ds * i # 掛け算のところ

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o) 
        dg *= (1 - g ** 2) # 自分のところに戻るようの逆伝播

        dA = np.hstack((df, dg, di, do)) # 結合

        dWh = np.dot(h_prev.T, dA) # 左端の逆伝播
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev # 3つ！，簡単です

class TimeLSTM:
    '''
    Time分出力できるやつ
    '''
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None: # statefulがFalseなら0にする
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None: # statefulがFalseなら0にする
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T): # 時間サイズをTへ
            layer = LSTM(*self.params) # 同じ重みを共有する
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None): # stateを消去
        self.h, self.c = h, c

    def reset_state(self): # 記憶セルもすべて消去
        self.h, self.c = None, None

class MatMul:
    def __init__(self, W):
        self.params = [W] # わざわざこうしてるのは，layersで処理するときに配列同士の足し算がうまくいかないから（layer毎の分離が出来なくなる）
        self.grads = [np.zeros_like(W)] # こちも同じ理由　レイヤー毎に分離したいので
        self.x = None

    def forward(self, x):
        # 順伝播
        W, = self.params # こうやると中身取り出せます
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        # 逆伝播
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW # deepコピーと同じ（アドレス固定する）pythonは値に割り振るのでそれを避ける
        return dx

class TimeAffine:
    '''
    AffineがT個分ある（行列演算レベルでくっつけてある）
    '''
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1) # 2次元に変換⇒次元だけは守っているイメージ，バッチっていう概念がなくなる感じ
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1) # 時系列データが出力される

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout) # こうすれば，横向きになっているから全部勾配が勝手に足される（forで回す必要がない）行×列でいける(D * N*T) * (N*H * H)かな
        dx = np.dot(dout, W.T) # こっちもおなじ原理
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2) # 何番目がもっとも大きいか（[]）の一番小さいところの行でみている

        # これなにやってんだろ
        mask = (ts != self.ignore_label) # -1なら排除してるんだけど，-1のときがあるっぽいな

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T) # indexです，列数
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts]) # 正解のindexだけ取り出している
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -1 * np.sum(ls)
        loss /= mask.sum() # mask部分だけ考える

        # print('mask = {0}'.format(mask.sum()))

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys # 出力をこっちにいれとく
        dx[np.arange(N * T), ts] -= 1 # one-hotの場合は，正解のところいがいtは0，しつこいけど行がバッチ！！！！
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx

class TimeDropout:
    '''
    ドロップアウト（Time用（つまり，すべての隠れ状態に対して行っている））
    '''
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio # このrationより大きい値だけをTrueにする，値はランダムにとっている
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype('f') * scale # マスクする

            return xs * self.mask # 大きいやつ以外が消える
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask # Relu関数的になっている，maskでTrueにしておけばよい，Trueのものだけ逆に伝わる

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


