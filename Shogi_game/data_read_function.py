# 棋譜を読み込むプログラム
import pickle
import sys
import numpy as np

# corpus作成用
def preprocess(kifu_data, word_to_id, id_to_word):
    for move in kifu_data:
        if move not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[move] = new_id
            id_to_word[new_id] = move

    corpus = np.array([word_to_id[w] for w in kifu_data])

    return corpus, word_to_id, id_to_word

class Kifu_info():
    def __init__(self, name, data_num):
        self.name = name
        self.data_num = data_num 

def load_kifu():
    '''
    トレーニング用の棋譜　読み込み
    '''
    # 読み込む名人の棋譜
    fujii = Kifu_info('fujii', 18)
    habu = Kifu_info('habu', 150)
    hifumin = Kifu_info('hifumin', 100)

    # test用の棋譜
    test = Kifu_info('test', 5)

    # 集めたいデータのクラスリスト
    names_and_num_list = [fujii, habu, hifumin, test]

    kifu_koko_data = []
    kifu_senko_data = []
    kifu_test_data = []

    for key in names_and_num_list:
        for i in range(key.data_num):
            # それぞれの棋譜
            each_kifu = ['開始'] # 開始文字を表すもの 

            f = open('./data/{0}/{1}.txt'.format(key.name, i+1))

            line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)

            while True:
                # いらない文字除外
                split_space = line.split() # 空欄
                
                if split_space[1] == '同': # 例外なので
                    split_space[1] = split_space[1] + split_space[2]

                split_kako = split_space[1].split('(') # カッコ
                move = split_kako[0]

                each_kifu.append(move)

                # print(move)

                # 投了ならpass
                if move == '投了':
                    break

                # 次を読み込み
                line = f.readline()
            
            # 連結
            if key is test:
                kifu_test_data.extend(each_kifu)
            else:
                # print(len(each_kifu))
                if len(each_kifu) % 2 == 0: # 後攻が勝った場合
                    kifu_koko_data.extend(each_kifu)
                    # print('kokowin = {0}'.format(np.array(kifu_koko_data)))
                    # sys.exit()
                else: # 先攻が勝った場合
                    kifu_senko_data.extend(each_kifu)
                    # print(np.array(kifu_senko_data))

            # print(np.array(kifu_data))
            # sys.exit()

            f.close()

    # print(np.array(kifu_data))
    # corpusを作成        
    word_to_id = {}
    id_to_word = {}
    senko_corpus, word_to_id, id_to_word = preprocess(kifu_senko_data, word_to_id, id_to_word)
    koko_corpus, word_to_id, id_to_word = preprocess(kifu_koko_data, word_to_id, id_to_word)
    corpus_test, word_to_id, id_to_word = preprocess(kifu_test_data, word_to_id, id_to_word)
    
    # print(word_to_id)
    # print(id_to_word)
    # print(corpus)
    # print(word_to_id)

    return senko_corpus, koko_corpus, corpus_test, word_to_id, id_to_word

if __name__ == '__main__':
    load_kifu()