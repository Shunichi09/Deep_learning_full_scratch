# 棋譜を読み込むプログラム
import pickle
import sys

class Datas():
    def __init__(self, name, data_num):
        self.name = name
        self.data_num = data_num 

fujii = Datas('fujii', 13)

names_and_num_list = [fujii]

kihu_data = []

for key in names_and_num_list:
    for i in range(key.data_num):
        f = open('./data/{0}/{1}.txt'.format(key.name, i+1))

        line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)

        while True:
            # いらない文字除外
            split_space = line.split() # 空欄
            
            if split_space[1] == '同': # 例外なので
                split_space[1] = split_space[1] + split_space[2]

            split_kako = split_space[1].split('(') # カッコ
            move = split_kako[0]

            print(move)

            # 投了ならpass
            if move == '投了':
                break

            # 次を読み込み
            line = f.readline()

        f.close()
        sys.exit()