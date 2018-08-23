import sys
sys.path.append('..')
from NN_ch6 import Rnnlm, BetterRnnlm# , RnnlmTrainer
from trainer_optimizer_ch6 import SGD, RnnlmTrainer
from common import ptb # dataset読み込む用
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions_ch6 import eval_perplexity

def main():
    # ハイパーパラメータの設定
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
    time_size = 35  # RNNを展開するサイズ
    lr = 20.0
    max_epoch = 4
    max_grad = 0.25

    # 学習データの読み込み
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    # モデルの生成
    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 勾配クリッピングを適用して学習
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
                eval_interval=20)
    trainer.plot(ylim=(0, 500))

    # テストデータで評価
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print('test perplexity: ', ppl_test)

    # パラメータの保存
    model.save_params()

if __name__ == '__main__':
    main()
