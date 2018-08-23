# chp4用の関数群
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

class UnigramSampler:
    '''
    negative sampleをとるクラス
    '''
    def __init__(self, corpus, power, sample_size):# corpusはコーパス，powerは0.75（全然出てこないやつの確率を多少上げる），samplesizeはいくつnegativesampleとるか
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter() # やる意味わからんけど辞書作成

        for word_id in corpus:
            counts[word_id] += 1 # 出現回数記録

        vocab_size = len(counts) # 語彙数
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size) # 出現確率
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power) # 0.75乗する 
        self.word_p /= np.sum(self.word_p) # 確率分布

    def get_negative_sample(self, target): # targetで正解データをもらえる
        batch_size = target.shape[0] # しつこいけど行がバッチ数に対応している！

        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32) # 型はint型でオッケー

        for i in range(batch_size): # バッチサイズ分だけターゲットがあるので
            p = self.word_p.copy()
            target_idx = target[i] # ターゲットがどのidなのかを取得
            p[target_idx] = 0 # そこの取得確率を0にする
            p /= p.sum() # sum，もう一回和をとって割り算
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p) # 確率分布に従って選ぶ（語彙数

        return negative_sample # idが変える # バッチサイズ×sample_sizeになる

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
