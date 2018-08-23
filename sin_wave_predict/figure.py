import matplotlib.pyplot as plt
import numpy as np

# plotクラス
class Formal_mul_ploter():
    def __init__(self, x, y, x_label_name, y_label_name, y_names):
        # figure作成
        self.fig = plt.figure(figsize=(8,6),dpi=100) # ここでいろいろ設定可能

        # subplot追加
        self.axis = self.fig.add_subplot(111)

        # dataの収納
        self.x = x
        self.y = y

        # legendの時に使う
        self.y_names = y_names

        # 最大値と最小値
        self.x_max = np.max(self.x)
        self.x_min = np.min(self.x)
        self.y_max = np.max(self.y)
        self.y_min = np.min(self.y)
        margin = 1

        # 軸（label_name）
        self.axis.set_xlabel(x_label_name)
        self.axis.set_ylabel(y_label_name)

        # 軸（equal系）
        # self.axis.axis('equal')

        # 軸（limit）# 調整可能
        # self.axis.set_xlim([self.x_min - margin, self.x_max + margin])
        # self.axis.set_ylim([self.y_min - margin, self.y_max + margin])

        # grid
        self.axis.grid(True)

        # color 好きなカラーリスト作っとく
        self.colors = ['b', 'r', 'k', 'm']

    def mul_plot(self):
        for i in range(len(self.y)):
            self.axis.plot(self.x, self.y[i], label=self.y_names[i], color=self.colors[i])

        self.axis.legend()

        plt.show()

if __name__ == '__main__':
    pass