import numpy as np
import matplotlib.pyplot as plt


def load_data():
    """加载数据并将x和y分离出来"""
    train = np.loadtxt("data/mnist_train_2000.csv", delimiter=",", dtype="int")
    test = np.loadtxt("data/mnist_test_200.csv", delimiter=",", dtype="int")

    x_train = train[:, 1:] / 255
    y_train = one_vs_all(train[:, 0])
    # print(x_train.shape)
    # print(y_train.shape)
    x_test = test[:, 1:] / 255
    y_test = test[:, 0]
    # print(x_test.shape)
    # print(y_test.shape)
    return x_train, y_train, x_test, y_test


def one_vs_all(y):
    """将y转化成多分类的形式[0 0 0 1 ... 0]"""
    y_temp = np.zeros([y.size, 10])
    for i in range(y.size):
        y_temp[i, y[i]] = 1
    return y_temp


class BPNeuralNetwork:
    def __init__(self, x_train, y_train, epochs=100, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.m, self.n = np.shape(x_train)
        self.layers = None
        self.loss_arr = []
        self.w = []
        self.d = []

    def sequential(self, layers):
        """存储层信息并初始化权重"""
        self.layers = layers
        length = self.n
        for i in range(len(self.layers)):
            self.w.append(np.random.randn(length, layers[i][0]) / np.sqrt(length))
            self.d.append(np.random.randn(layers[i][0]) / np.sqrt(length))
            length = layers[i][0]

    def dense(self, units, activation="tanh"):
        """传递层信息"""
        return [units, activation]

    def forward(self):
        """前向传播"""
        z = []
        a = [self.x_train]
        for i in range(len(self.layers)):
            z.append(a[i] @ self.w[i] + self.d[i])
            if self.layers[i][1] == 'sigmoid':
                a.append(self.sigmoid(z[i]))
            elif self.layers[i][1] == 'tanh':
                a.append(self.sigmoid(z[i]))
            else:
                raise Exception("没有%s这个激活函数" % self.layers[i][1])
        return z, a

    def back(self, z, a):
        """反向传播
        grad_w(i) = l(i) * x(i)
        l_i = l(i+1) * w(i+1) * derivative_f(z(i))
        l_last = (y_hat - y) * derivative_f(z(last))
        """
        grad_w = [0] * len(self.layers)
        grad_d = [0] * len(self.layers)
        # 先计算最后一层的梯度
        if self.layers[-1][1] == 'sigmoid':
            l = (a[-1] - self.y_train) * self.derivative_sigmoid(z[-1])
        elif self.layers[-1][1] == 'tanh':
            l = (a[-1] - self.y_train) * self.derivative_tanh(z[-1])
        rows, columns = l.shape
        grad_w[-1] = a[-2].T @ l / rows
        grad_d[-1] = np.sum(l, axis=0) / rows
        for i in range(len(self.layers) - 2, -1, -1):
            if self.layers[i][1] == 'sigmoid':
                l = (l @ self.w[i+1].T) * self.derivative_sigmoid(z[i])
            elif self.layers[i][1] == 'tanh':
                l = (l @ self.w[i+1].T) * self.derivative_tanh(z[i])
            rows, columns = l.shape
            grad_w[i] = a[i].T @ l / rows
            grad_d[i] = np.sum(l, axis=0) / rows
        return grad_w, grad_d

    def fit(self):
        for epoch in range(self.epochs):
            z, a = self.forward()  # 正向传播
            grad_w, grad_d = self.back(z, a)  # 反向传播
            # 梯度下降
            for i in range(len(self.layers)):
                self.w[i] = self.w[i] - self.learning_rate * grad_w[i]
                self.d[i] = self.d[i] - self.learning_rate * grad_d[i]

            loss = self.loss(a[-1], self.y_train)
            self.loss_arr.append(loss)
            print("--------------------第%d次迭代" % (epoch + 1))
            print("loss:", loss)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative_sigmoid(self, z):
        return np.exp(-z) / np.square(1 + np.exp(-z))

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def derivative_tanh(self, z):
        return 4.0 / np.square(np.exp(z) + np.exp(-z))

    def loss(self, y_hat, y):
        rows, columns = y.shape
        return np.sum(np.square(y_hat - y)) / (2 * columns)

    def loss_graph(self):
        x = [i for i in range(self.epochs)]
        plt.figure(figsize=(6, 4))
        plt.plot(x, self.loss_arr, linewidth=1)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss")
        plt.savefig('loss_BP.png', dpi=120, bbox_inches='tight')
        plt.show()

    def predict(self, x_test, y_test):
        a = x_test
        for i in range(len(self.layers)):
            z = a @ self.w[i] + self.d[i]
            if self.layers[i][1] == "sigmoid":
                a = self.sigmoid(z)
            elif self.layers[i][1] == "tanh":
                a = self.tanh(z)
        location = np.argmax(a, axis=1)
        length = y_test.size
        correct = [1 if location[i] == y_test[i] else 0 for i in range(length)]
        success = sum(correct)
        print("预测数据总数%d" % length)
        print("预测正确的数据个数%d" % success)
        print("正确率:{:.2%}".format(success / length))


if __name__ == "__main__":
    load_data()
    x_train, y_train, x_test, y_test = load_data()

    model = BPNeuralNetwork(x_train, y_train, epochs=3000, learning_rate=0.8)
    model.sequential([model.dense(32, activation='tanh'),
                      model.dense(10, activation='sigmoid')])
    model.fit()
    model.loss_graph()
    model.predict(x_test, y_test)
