# 標準ライブラリ系
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import copy

# 関数
from functions_sin import sigmoid

class RNN():
    '''
    RNNの1ステップの処理を行うレイヤーの実装
    '''
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b] # くくっているのは同じ理由
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev): # h_prevは1つ前の状態
        Wx, Wh, b = self.params # 取り出す
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b

        h_next = np.tanh(t) # tanh関数

        self.cache = (x, h_prev, h_next) # 値を保存しておく
        
        return h_next
    
    def backward(self, dh_next): # 隠れ層の逆伝播の値が引数
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2) # tanhの逆伝播（各要素に対してかかる）
        db = np.sum(dt, axis=0) # いつものMatmulと同じ
        dWh = np.dot(h_prev.T, dt) # いつものMatmulと同じ
        dh_prev = np.dot(dt, Wh.T) # 上の式みて考えれば分かる
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx # 値をコピー
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN:
    '''
    上のやつ全部まとめたやつBPTTさせる分
    '''
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b] # くくっているのは同じ理由 hWh + xWx + b = h
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.T = None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params # パラメータの初期化
        N, self.T, D = xs.shape # xsの形, Dは入力ベクトルの大きさ，このレイヤーはまとめてデータをもらうので！

        D, H = Wx.shape 

        self.layers = [] # 各レイヤー（RNNの中の）
        hs = np.empty((N, self.T, H), dtype='f') # Nはバッチ数，Tは時間数，HがHの次元

        if not self.stateful or self.h is None: # statefulでなかったら,または，初期呼び出し時にhがなかったら（前の状態を保持しなかったら）
            self.h = np.zeros((N, H), dtype='f') # Nはバッチ数

        for t in range(self.T): # 時間分（backpropする分）だけ繰り返し
            layer = RNN(*self.params) # 可変長引数らしい　ばらばらで渡される今回のケースでいえば，Wx, Wh, bとしても同義
            self.h = layer.forward(xs[:, t, :], self.h) # その時刻のxを渡す
            hs[:, t, :] = self.h # 保存しておく
            self.layers.append(layer) # RNNの各状態の保存

        # 出力はhsの最後のものだけ
        hs = hs[:, -1, :]

        # print('hs= {0}'.format(hs))
        # a = input()

        return hs

    def backward(self, dhs): 
        Wx, Wh, b = self.params # パラメータの初期化
        N, H = dhs.shape # xsの形, Dは入力ベクトルの大きさ，このレイヤーはまとめてデータをもらうので！
        D, H = Wx.shape 

        dxs = np.empty((N, self.T, D), dtype='f')
        grads = [0, 0, 0]

        for t in reversed(range(self.T)):
            layer = self.layers[t] # 一つずつ保存しておいたlayerを呼び出す
            dx, dhs = layer.backward(dhs) 
            dxs[:, t ,:] = dx

            for i, grad in enumerate(layer.grads): # 各重み(3つ，Wx, Wb, b)を取り出す，同じ重みを使っているので，勾配はすべて足し算
                grads[i] += grad 
        
        # print(len(grads))

        for i, grad in enumerate(grads): # 時系列順に並んでいるやつをコピー
            self.grads[i][...] = grad # 
        
        self.dh = dhs

        return None# dxs # 後ろに逆伝播させる用(N, T, D)になっている

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
        N, H = h_prev.shape # 隠れ状態のサイズ，batch×大きさ

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
        self.T = None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, self.T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, self.T, H), dtype='f')

        if not self.stateful or self.h is None: # statefulがFalseなら0にする
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None: # statefulがFalseなら0にする
            self.c = np.zeros((N, H), dtype='f')

        for t in range(self.T): # 時間サイズをTへ
            layer = LSTM(*self.params) # 同じ重みを共有する
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h # RNNの各状態の保存

            self.layers.append(layer)

        # 出力はhsの最後のものだけ
        hs = hs[:, -1, :]

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, H = dhs.shape # 時刻にして1つ分しか返ってこないはず
        D = Wx.shape[0]

        dxs = np.empty((N, self.T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(self.T)):
            layer = self.layers[t]
            dx, dhs, dc = layer.backward(dhs, dc)
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

class TimeAffine:
    '''
    AffineがT個分ある（行列演算レベルでくっつけてある）
    '''
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, D = x.shape
        W, b = self.params

        rx = x
        out = np.dot(rx, W) + b
        self.x = x
        return out # 時系列データが出力される

    def backward(self, dout):
        x = self.x
        N, D = x.shape
        W, b = self.params

        rx = x

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout) # こうすれば，横向きになっているから全部勾配が勝手に足される（forで回す必要がない）行×列でいける(D * N*T) * (N*H * H)かな
        dx = np.dot(dout, W.T) # こっちもおなじ原理

        self.grads[0][...] = dW
        self.grads[1][...] = db

        # print('dx= {0}'.format(dx))
        # a = input()

        return dx

class TimeIdentifyWithLoss:
    '''
    時系列データをまとめて受け付ける損失関数
    '''
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.counter = 0

    def forward(self, xs, ts):
        N, D = xs.shape # ここでDは1

        ys = copy.deepcopy(xs) # 恒等関数

        # print('ts = {0}'.format(ts))
        
        loss = 0.5 * np.sum((ys - ts)**2)
        loss /= N # 1データ分での誤差

        # print('Y = {0}, T = {1}'.format(np.round(ys, 3), np.round(ts, 3)))
        # print('N * T = {0}'.format(N*T))
        # print('loss = {0}'.format(loss))
        # if self.counter % 1 == 0:
            # plt.plot(range(len(ys.flatten())) , ys.flatten())
            # plt.plot(range(len(ys.flatten())) , ts.flatten())
            # plt.show()
        # a = input()

        self.cache = (ts, ys, (N, D))
        self.counter += 1

        return loss

    def backward(self, dout=1):
        ts, ys, (N, D) = self.cache

        dx = ys - ts # 出力をこっちにいれとく
        dx /= N

        return dx