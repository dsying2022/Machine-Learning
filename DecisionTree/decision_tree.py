import numpy as np
import matplotlib.pyplot as plt
from math import log
import random


def load_data():
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


def split_dataset(features, labels, split_rate=0.7):
    length = len(labels)
    x_train = features[:int(length*split_rate)]
    x_test = features[int(length*split_rate):]
    y_train = labels[:int(length * split_rate)]
    y_test = labels[int(length * split_rate):]
    return x_train, y_train, x_test, y_test


class DecisionTree:
    def entropy(self, y):
        """计算信息熵"""
        label_num = {}
        for i in y:
            if i not in label_num.keys():
                label_num[i] = 1
            else:
                label_num[i] += 1
        ent = 0
        for i in label_num:
            temp = float(label_num[i]) / len(y)
            ent -= temp * log(temp, 2)
        return ent

    def split_set(self, x, y, pos, value):
        """按照节点划分左右集合"""
        x_left = []
        x_right = []
        y_left = []
        y_right = []
        for index, sample in enumerate(x):
            if sample[pos] < value:
                x_left.append(sample)
                y_left.append(y[index])
            else:
                x_right.append(sample)
                y_right.append(y[index])
        return x_left, y_left, x_right, y_right

    def node_feature(self, x, y):
        """选出最佳分离的节点"""
        current_ent = self.entropy(y)
        best_gain = 0
        best_feature = [-1, -1]
        m, n = np.shape(x)
        for i in range(n):
            feature = [j[i] for j in x]
            unique = set(feature)
            for value in unique:
                x_left, y_left, x_right, y_right = self.split_set(x, y, i, value)
                # 计算信息增益
                temp = len(y_left) / float(len(y))
                new_ent = temp * self.entropy(y_left)
                new_ent += (1 - temp) * self.entropy(y_right)
                gain = current_ent - new_ent
                if gain > best_gain:
                    best_gain = gain
                    best_feature = [i, value]
        return best_gain, best_feature

    def create_tree(self, x, y):
        """利用递归方法创建决策树"""
        tree = {}
        label_set = set(y)
        tree['sample_num'] = len(y)
        if len(label_set) == 1:  # 终止条件是一个集合中全是相同的类
            tree['class'] = y[0]
            return tree
        best_gain, best_feature = self.node_feature(x, y)  # 选出最佳分离的节点 输出分别为信息增益和用于分离的特征
        tree['node'] = best_feature
        x_left, y_left, x_right, y_right = self.split_set(x, y, best_feature[0], best_feature[1])  # 开始划分左右集合
        tree['left'] = self.create_tree(x_left, y_left)  # 左边重复上述操作
        tree['right'] = self.create_tree(x_right, y_right)  # 右边重复上述操作
        return tree

    def predict(self, x, y, tree):
        y_hat = []
        for sample in x:
            temp = tree
            while 'class' not in temp.keys():
                if sample[temp['node'][0]] < temp['node'][1]:
                    temp = temp['left']
                else:
                    temp = temp['right']
            y_hat.append(temp['class'])
        correct_num = 0
        for i in range(len(y)):
            if y[i] == y_hat[i]:
                correct_num += 1
        print("待预测数据总个数:%d" % len(y))
        print("预测正确的个数:%d" % correct_num)
        print("准确率:{:0.2%}".format(correct_num / len(y)))


if __name__ == "__main__":
    print("------------加载数据--------------")
    features_name, features, labels = load_data()
    print("数据加成功")
    print("------------划分数据集------------")
    x_train, y_train, x_test, y_test = split_dataset(features, labels)
    print("数据集划分成功")
    print("----------创建决策树模型-----------")
    model = DecisionTree()
    tree = model.create_tree(x_train, y_train)
    print("模型创建成功，输出得到的决策树")
    print(tree)
    print("------------模型评估-------------")
    model.predict(x_test, y_test, tree)
