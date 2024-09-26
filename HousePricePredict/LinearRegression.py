# -*- coding: utf-8 -*-
"""
@file: LinearRegression.py
@time: 2023/5/28 0:47
@author: ShuZhiMin
"""
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    train_x = []
    train_y = []
    with open("data1.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(',')
            train_x.append(int(line[0]) / 1000)
            train_y.append(int(line[2]) / 1000)

    return train_x, train_y


def graph(w, b, x, y):
    lr_x = np.linspace(min(x)-0.5, max(x)+0.5, 50)
    lr_y = lr_x * w + b
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, 'o', linewidth=1)
    plt.plot(lr_x, lr_y, linewidth=1)
    plt.xlabel("1000 sqft")
    plt.ylabel("1000s of dollars")
    plt.title("LinearRegressionVisualization\n w=%f b=%f" % (w, b))
    plt.savefig('LinearRegressionVisualization.png', dpi=120, bbox_inches='tight')
    plt.show()

class LinearRegression:
    def __init__(self, train_x, train_y, iteration=100, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.iteration = iteration
        self.loss = []
        self.m = len(train_x)
        self.w = 0
        self.d = 0

    def cost(self):
        loss = np.sum(np.square(self.w * self.train_x + self.d - self.train_y)) / (2 * self.m)
        return loss

    def grad(self):
        grad_w = np.dot(self.w * self.train_x + self.d - self.train_y, self.train_x) / self.m
        grad_d = np.sum(self.w * self.train_x + self.d - self.train_y) / self.m
        return grad_w, grad_d

    def train(self):
        for i in range(self.iteration):
            grad_w, grad_d = self.grad()
            self.w = self.w - self.learning_rate * grad_w
            self.d = self.d - self.learning_rate * grad_d
            cost = self.cost()
            self.loss.append(cost)
            print("--------------------第%d次迭代" % (i+1))
            print("loss:", cost)
        return self.w, self.d

    def loss_graph(self):
        x = [i for i in range(self.iteration)]
        plt.figure(figsize=(6, 4))
        plt.plot(x, self.loss, linewidth=1)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss")
        plt.savefig('loss_LinearRegression.png', dpi=120, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    train_x, train_y = load_data()
    # print(train_x)
    # print(train_y)
    LR = LinearRegression(train_x, train_y, iteration=100, learning_rate=0.01)
    # print(LR.cost())
    # print(LR.m)
    w, d = LR.train()
    LR.loss_graph()
    np.savetxt('Result_LinearRegression.txt', np.append(w, d), fmt='%f')
    graph(w, d, train_x, train_y)


