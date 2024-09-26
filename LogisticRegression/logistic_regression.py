# -*- coding: utf-8 -*-
"""
@file: logistic_regression.py
@time: 2023/5/30 19:20
@author: ShuZhiMin
"""
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    x_train = []
    y_train = []
    with open("ex2data1.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n').split(',')
            x_temp = []
            n = len(line)
            for i in range(n - 1):
                x_temp.append(float(line[i]))
            x_train.append(x_temp)
            y_train.append(int(line[-1]))
    return x_train, y_train


'''回归曲线可视化'''
def graph(x, y, w, d, polynomial_n, x_polynomial):
    n = 100
    x_0 = x[np.where(y == 0), :].reshape(-1, 2)
    x_1 = x[np.where(y == 1), :].reshape(-1, 2)
    plt.scatter(x_0[:, 0], x_0[:, 1])
    plt.scatter(x_1[:, 0], x_1[:, 1])

    u = np.linspace(min(x[:, 0]), max(x[:, 0]), n)
    v = np.linspace(min(x[:, 1]), max(x[:, 1]), n)
    z = np.zeros((u.size, v.size))  # 网格化

    for k in range(n):
        for m in range(n):
            x_temp = [u[k], v[m]]
            for i in range(2, polynomial_n + 1):
                for j in range(i + 1):
                    x_temp = np.hstack((x_temp, np.power(x_temp[0], i - j)*np.power(x_temp[1], j)))

            x_temp = (x_temp - np.mean(x_polynomial, axis=0)) / (np.max(x_polynomial, axis=0) - np.min(x_polynomial, axis=0))
            z[k, m] = np.dot(x_temp, w) + d

    plt.contour(u, v, z, 0, colors='b')
    plt.title('Decision Boundary')
    plt.savefig('Decision_Boundary_1.png', dpi=120, bbox_inches='tight')
    plt.show()


class logistic_regression:
    def __init__(self, x_train, y_train, polynomial_n=1, iteration=100, learning_rate=0.01, lambda_=1):
        self.polynomial_n = polynomial_n
        self.lambda_ = lambda_  # 正则化参数
        self.learning_rate = learning_rate
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.iteration = iteration
        self.loss = []
        self.m, self.n = self.x_train.shape
        self.w = np.zeros(self.n)
        self.d = 0

    def polynomial_data(self):
        for i in range(2, self.polynomial_n + 1):
            for j in range(i + 1):
                self.x_train = np.hstack((self.x_train,
                                          np.multiply(np.power(self.x_train[:, 0], i - j),
                                                      np.power(self.x_train[:, 1], j)).reshape(-1, 1)))
        self.m, self.n = self.x_train.shape
        self.w = np.zeros(self.n)
        return self.x_train

    def MeanNormalization(self):
        self.x_train = (self.x_train - np.mean(self.x_train, axis=0)) / (np.max(self.x_train, axis=0) - np.min(self.x_train, axis=0))

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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost(self):
        f = self.sigmoid(np.dot(self.x_train, self.w) + self.d)
        first = np.dot(-self.y_train, np.log(f))
        second = np.dot((1 - self.y_train), np.log(1 - f))
        loss = np.sum(first - second) / self.m + np.sum(np.square(self.w)) * self.lambda_ / self.m
        return loss

    def grad(self):
        f = self.sigmoid(np.dot(self.x_train, self.w) + self.d)
        grad_w = np.dot(f - self.y_train, self.x_train) / self.m
        grad_d = np.sum(f - self.y_train) / self.m
        return grad_w, grad_d

    def loss_graph(self):
        x = [i for i in range(self.iteration)]
        plt.figure(figsize=(6, 4))
        plt.plot(x, self.loss, linewidth=1)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss")
        plt.savefig('loss_LogisticRegression_1.png', dpi=120, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    x_train, y_train = load_data()
    lr = logistic_regression(x_train, y_train, polynomial_n=3, iteration=3000, learning_rate=0.3, lambda_=0.01)
    x_polynomial = lr.polynomial_data()
    lr.MeanNormalization()
    w, d = lr.train()
    lr.loss_graph()
    graph(np.array(x_train), np.array(y_train), w, d, lr.polynomial_n, x_polynomial)
    np.savetxt('Result_LogisticRegression_1.txt', np.append(w, d), fmt='%f')

