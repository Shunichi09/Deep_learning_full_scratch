# 標準ライブラリ系
import numpy as np
import collections
import sys

def softmax(x):
    if x.ndim == 2: 
        x = x - x.max(axis=1, keepdims=True) # 形を守ったまま最大で構成して引き算する（計算結果がでないのを防ぐ）
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True) # これも形を守ったまま和を計算して各要素を割り算する
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def cross_entropy_error(y, t):
    '''
    出力はバッチ×出力数で出てくるはず
    それを受け取ってlossを計算
    '''
    if y.ndim == 1: # 1次元のとき（ばっちが1つのときってこと）行ベクトルに変形
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)# 1をとっているとこを抽出（方向は行で）
             
    batch_size = y.shape[0]
    # エラーをとるのは，one-hot-vectorの正解部分だけです，マイナスとるのはもともとが小数点の話（softmaxの関係で）なので，マイナスになってしまうからです！
    return -1 * np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # batchsizeで割り算して平均取ってます

def sigmoid(x):
    '''
    sigmoid関数
    '''
    return 1 / (1 + np.exp(-x))


def relu(x):
    '''
    relu関数
    '''
    return np.maximum(0, x)


def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('evaluating perplexity ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl

