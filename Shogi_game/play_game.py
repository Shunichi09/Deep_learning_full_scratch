# 標準ライブラリ系
import sys
sys.path.append('..')
import numpy as np
import math

# 関数系
from NNs_shogi import BetterRnnlmGen, RnnlmGen
from data_read_function import load_kifu

corpus, corpus_test, word_to_id, id_to_word = load_kifu()
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen(vocab_size=vocab_size, wordvec_size=650, hidden_size=650, dropout_ratio=0.0)
# model = RnnlmGen()

model.load_params('BetterRnnlm.pkl')

print('先攻=1ですか？後攻=0ですか？，数字は半角で入力')
my_turn_flag = int(input())

if my_turn_flag:
    text = 'という手を提案します'
else: 
    text = 'という手を打ってくると思います'

print('８四歩，のように打ってください，棋譜の数字は全角です \n')
print('一番初めは，開始と打ってください \n')

# 対戦の棋譜が保存されます
moves = []

while True:
    move = input()
    moves.append(move)

    if move == '投了':
        print('game is done')
        print('棋譜一覧です　{0}' .format(np.array(moves)))
        break

    # idに変換
    word_id = word_to_id[move]

    # 候補が返ってくる
    candidate_move_ids, candidate_logit = model.generate(word_id, sample_size=1)

    for i, word_id in enumerate(candidate_move_ids):
        logit = round(float(candidate_logit[i]), 2)
        print(' {0} ％ で，{1}　{2}'.format(logit * 100, id_to_word[word_id], text))

    if my_turn_flag:
        print('あなたのターンです（上記から実際に打った手を入力してください）')
        my_turn_flag = False
        text = 'という手を打ってくると思います'
    else: 
        print('相手のターンです（上記から実際に打たれた手を入力してください）')
        my_turn_flag = True
        text = 'という手を提案します'

