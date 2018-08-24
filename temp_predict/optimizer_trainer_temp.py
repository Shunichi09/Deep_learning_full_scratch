# 標準ライブラリ系
import numpy as np
import time
import matplotlib.pyplot as plt

class Trainer:
    '''
    通常で学習させる方法
    '''
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, time_size=1, max_epoch=10, batch_size=32, max_grad=None, eval_interval=10):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # シャッフル
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size].reshape(batch_size, time_size, model.input_size)
                batch_t = t[iters*batch_size:(iters+1)*batch_size].reshape(batch_size, model.output_size)

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

            # エポックごとの評価
            elapsed_time = time.time() - start_time
            ave_loss = total_loss / loss_count
            print('Time {0}| epoch {1} | ave_loss {2}'.format(round(elapsed_time, 1), epoch, round(ave_loss, 5)))
            self.loss_list.append(round(ave_loss, 5))
            total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

class RnnlmTrainer:
    '''
    RNN手法で学習させる方法
    '''
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        # ミニバッチの取得
        batch_x = np.empty((batch_size, time_size, self.model.input_size), dtype='f')
        batch_t = np.empty((batch_size, self.model.input_size), dtype='f')
        num = np.empty((batch_size, time_size, self.model.input_size), dtype='f')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # バッチの各サンプルの読み込み開始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                num[i, time] = (offset + self.time_idx) % data_size

                if time == time_size-1:
                    batch_t[i, :] = t[(offset + self.time_idx) % data_size] # 最後のデータのみ

            # print(num[i])
            # print(batch_x[0])
            # print(batch_t[0])
            # a = input()
            
            self.time_idx += 1

        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        # print(len(xs))

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

            # エポックごとの評価
            elapsed_time = time.time() - start_time
            ave_loss = total_loss / loss_count
            print('Time {0}| epoch {1} | ave_loss {2}'.format(round(elapsed_time, 1), epoch, round(ave_loss, 5)))
            self.loss_list.append(round(ave_loss, 5))
            total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('rplexity')
        plt.show()

class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    加算するのは，今回でいえば共有するものが2つあるからって感じ
    誤差自体というか傾きは重みを共有している分だけ変化することになる（共有するってことはそういうこと），それぞれで更新されるので
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params) #

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]: # 何番目のレイヤー同士が一緒かをみてる・ここで重みが同じか判定(is 演算子はオブジェクトが同一か判定します)  == は値
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j) # 取り除く（レイヤーの集合から）
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break # popするので，これでよい（各要素の最初の部分だけみることになるけど）
            if find_flg: break

        if not find_flg: break # 共通部分がない場合

    return params, grads

def clip_grads(grads, max_norm):
    '''
    勾配クリッピング
    RNNで使います
    ある勾配より大きくなったらもう使わないってやつ
    '''
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
