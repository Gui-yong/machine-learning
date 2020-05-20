import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Perceptron:
    def __init__(self, x, y, lr=0.1):
        self.x = x
        self.y = y
        self.lr = lr
        self.w = np.random.uniform(0, 1, size=(x.shape[1]))  # 初始化w
        self.b = 0  # 初始化b
        self.done = False

    def train(self,):
        while not self.done:
            index = -1
            for i in range(self.x.shape[0]):
                if self.y[i] != sign(np.dot(self.w, self.x[i]) + self.b):
                    index = i
                    break
            if index == -1:
                break
            self.w += self.lr * self.y[index] * self.x[index]
            self.b += self.lr * self.y[index]
        return self.w, self.b


if __name__ == '__main__':
    data = pd.read_csv("./train_data.csv")
    train_x = data.loc[:, ["x1", "x2"]].values
    train_y = data.loc[:, "y"].values
    class_1 = np.where(train_y > 0)
    class_2 = np.where(train_y < 0)

    perceptron = Perceptron(train_x, train_y)
    w, b = perceptron.train()

    line_x = np.arange(0, 10, 1)
    line_y = - (w[0] * line_x + b) / w[1]

    plt.scatter(train_x[class_1][:, 0], train_x[class_1][:, 1], c="red")
    plt.scatter(train_x[class_2][:, 0], train_x[class_2][:, 1], c="blue")
    plt.plot(line_x, line_y, "black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
