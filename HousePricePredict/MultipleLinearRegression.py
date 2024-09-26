# -*- coding: utf-8 -*-
"""
@file: MultipleLinearRegression.py
@time: 2023/5/28 10:13
@author: ShuZhiMin
"""
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    train_x = []
    train_y = []
    with open("data2.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(',')
            x_temp = []
            n = len(line)
            for i in range(n-1):
                x_temp.append(float(line[i]))
            train_x.append(x_temp)
            train_y.append(float(line[-1]))

    return train_x, train_y


class MultipleLinearRegression:
    def __init__(self, train_x, train_y, iteration=100, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.iteration = iteration
        self.loss = []
        self.m, self.n = self.train_x.shape
        self.w = np.zeros(self.n)
        self.d = 0

    def MeanNormalization(self):
        self.train_x = (self.train_x - np.mean(self.train_x, axis=0)) / (np.max(self.train_x, axis=0) - np.min(self.train_x, axis=0))
        self.train_y = (self.train_y - np.mean(self.train_y)) / (np.max(self.train_y) - np.min(self.train_y))

    def train(self):
        for i in range(self.iteration):
            grad_w, grad_d = self.grad()
            self.w = self.w - self.learning_rate * grad_w
            self.d = self.d - self.learning_rate * grad_d

            cost = self.cost()
            self.loss.append(cost)
            print("--------------------第%d次迭代" % (i + 1))
            print("loss:", cost)
        return self.w, self.d

    def grad(self):
        grad_w = np.dot(np.dot(self.train_x, self.w) + self.d - self.train_y, self.train_x) / self.m
        grad_d = np.sum(np.dot(self.train_x, self.w) + self.d - self.train_y) / self.m
        return grad_w, grad_d

    def cost(self):
        loss = np.sum(np.square(np.dot(self.train_x, self.w) + self.d - self.train_y)) / (2 * self.m)
        return loss

    def loss_graph(self):
        x = [i for i in range(self.iteration)]
        plt.figure(figsize=(6, 4))
        plt.plot(x, self.loss, linewidth=1)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss")
        plt.savefig('loss_MultipleLinearRegression.png', dpi=120, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    train_x, train_y = load_data()
    MLR = MultipleLinearRegression(train_x, train_y, iteration=1000, learning_rate=0.3)
    MLR.MeanNormalization()
    w, d = MLR.train()
    MLR.loss_graph()
    np.savetxt('Result_MultipleLinearRegression.txt', np.append(w, d), fmt='%f')
    # print(MLR.train_x)
    # print(MLR.train_y)
