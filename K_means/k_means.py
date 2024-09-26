import numpy as np
import matplotlib.pyplot as plt
from math import log
import random


def load_data():
    """加载数据"""
    features = []
    labels = []
    with open("iris.csv", "r") as f:
        lines = f.readlines()
        features_name = lines[0].split(",")[:-1]
        lines = lines[1:]
        random.shuffle(lines)
        for line in lines:
            temp = line.strip("\n").split(",")
            features.append(temp[:-1])
            labels.append(temp[-1])
    features = np.array(features, dtype="float")
    return features_name, features, labels


class Kmeans:
    def __init__(self, x, init_random_num=100, epochs=10, class_num=3):
        self.x = x
        self.init_random_num = init_random_num  # 进行聚类中心初始化的随机次数
        self.epochs = epochs  # 更新次数
        self.class_num = class_num  # 分几类
        self.cluster_center = np.zeros(class_num)  # 存储聚类中心
        self.row = len(x)
        self.C = np.zeros(self.row, dtype=int)  # 存储每个样本所对应的类
        self.loss_arr = []  # 存储每次更新的loss

    def init_cluster_center(self):
        """初始化聚类中心"""
        min_loss = np.Inf
        temp = self.cluster_center
        for i in range(self.init_random_num):
            self.cluster_center = random.sample(list(self.x), self.class_num)
            self.classify()
            loss_value = self.loss()
            if loss_value < min_loss:
                min_loss = loss_value
                temp = self.cluster_center
        self.loss_arr.append(min_loss)
        self.cluster_center = temp

    def distance(self, x0, x1):
        """计算样本到中心点的距离"""
        return np.sum(np.square(x0 - x1))

    def classify(self):
        """根据聚类中心进行分类"""
        for index, sample in enumerate(self.x):
            min_distance = np.Inf
            flag = 0
            for j in range(self.class_num):
                d = self.distance(sample, self.cluster_center[j])
                if d < min_distance:
                    min_distance = d
                    flag = j
            self.C[index] = flag

    def update_cluster_center(self):
        """根据分好类的集合更新聚类中心"""
        for i in range(self.class_num):
            self.cluster_center[i] = np.mean(self.x[np.where(self.C == i)[0], :], axis=0)

    def loss(self):
        """计算更新后的损失"""
        loss_value = 0
        for index, sample in enumerate(self.x):
            loss_value += self.distance(sample, self.cluster_center[self.C[index]])
        loss_value = loss_value / self.row
        return loss_value

    def train(self):
        """"训练模型"""
        for epoch in range(self.epochs):
            self.classify()  # 分类
            self.update_cluster_center()  # 更新
            loss_value = self.loss()  # 计算损失
            self.loss_arr.append(loss_value)
            print("第%d次迭代的loss:%f" % (epoch+1, loss_value))
        return self.C

    def loss_graph(self):
        """画随着每次epoch变化的loss图像"""
        x = [i for i in range(self.epochs + 1)]
        plt.figure(figsize=(6, 4))
        plt.plot(x, self.loss_arr, linewidth=1)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss")
        plt.savefig('loss_kmeans.png', dpi=120, bbox_inches='tight')
        plt.show()


def accuracy(labels, class_index):
    """评估模型，计算模型的准确率"""
    labels = np.array(labels)
    setosa_index = np.where(labels == 'setosa')
    versicolor_index = np.where(labels == 'versicolor')
    virginica_index = np.where(labels == 'virginica')
    setosa_class = class_index[setosa_index]
    versicolor_class = class_index[versicolor_index]
    virginica_class = class_index[virginica_index]
    all_class = [setosa_class, versicolor_class, virginica_class]
    correct_num = 0
    all_num = 0
    for i in range(3):
        counts = [0, 0, 0]
        for j in all_class[i]:
            if j == 0:
                counts[0] += 1
            elif j == 1:
                counts[1] += 1
            elif j == 2:
                counts[2] += 1
        all_num += sum(counts)
        correct_num += max(counts)
        print("%d号类总数:%d,预测正确的个数:%d" % (i, sum(counts), max(counts)))
    print("准确率:{:0.2%}".format(correct_num / all_num))


if __name__ == "__main__":
    features_name, features, labels = load_data()
    model = Kmeans(features, init_random_num=100, epochs=20, class_num=3)
    model.init_cluster_center()
    class_index = model.train()
    model.loss_graph()
    accuracy(labels, class_index)
