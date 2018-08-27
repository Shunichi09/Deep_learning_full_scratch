# 標準ライブラリ系
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions_shogi import eval_perplexity

# 関数等その他もろもろ
from NNs_shogi import Rnnlm, BetterRnnlm
from trainer_optimizer_shogi import SGD, RnnlmTrainer
from data_read_function import load_kifu

def main():
    # ハイパーパラメータの設定
    batch_size = 3
    wordvec_size = 650
    hidden_size = 650
    time_size = 35
    lr = 20.0
    max_epoch = 40
    max_grad = 0.25
    dropout = 0.5

    # 将棋データの読み込み
    senko_corpus, koko_corpus, corpus_test, word_to_id, id_to_word = load_kifu()

    save_data_name = ['senko_shogiRNN_param.pkl', 'koko_shogiRNN_param.pkl']

    vocab_size = len(word_to_id)
    xs = [senko_corpus[:-1], koko_corpus[:-1]]
    ts = [senko_corpus[1:], koko_corpus[1:]]

    for i in range(2):
        # モデルの生成（先攻用と後攻用）
        model = BetterRnnlm(vocab_size=vocab_size, wordvec_size=wordvec_size, hidden_size=hidden_size, dropout_ratio=dropout)
        optimizer = SGD(lr)
        trainer = RnnlmTrainer(model, optimizer)

        # 勾配クリッピングを適用して学習
        trainer.fit(xs[i], ts[i], max_epoch, batch_size, time_size, max_grad,
                    eval_interval=20)
        trainer.plot()

        # テストデータで評価
        model.reset_state()
        ppl_test = eval_perplexity(model, corpus_test)
        print('test perplexity: ', ppl_test)

        # パラメータの保存
        model.save_params(save_data_name[i])

    plt.show()

if __name__ == '__main__':
    main()