# 標準ライブラリ系
import sys
sys.path.append('..')
import numpy as np

# 関数系
from NNs_shogi import BetterRnnlmGen, RnnlmGen
from data_read_file import load_kifu

corpus, corpus_test, word_to_id, id_to_word = load_kifu()
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen(vocab_size=vocab_size, wordvec_size=650, hidden_size=650, dropout_ratio=0.0)
# model = RnnlmGen()

model.load_params('BetterRnnlm.pkl')

print('開始と打ってください')
print('また，８四歩，のように打ってください，数字は全角です')

moves = []

while True:
    move = input()
    moves.append(move)

    if move == '投了':
        print('game is done')
        break

    start_ids = [word_to_id[w] for w in moves]

    for x in start_ids[:-1]:
        x = np.array(x).reshape(1, 1)
        model.predict(x)

    # 候補が返ってくる
    candidate_move_ids, candidate_logit = model.generate(start_ids[-1], sample_size=1)

    # print('a  = {0}'.format(candidate_move_ids))

    for i, word_id in enumerate(candidate_move_ids):
        logit = round(candidate_logit[i], 3)
        print('{0} ％ で，{1}　という手が良いです \n'.format(logit, id_to_word[word_id]))
