import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_train():
    df = pd.read_csv('rating_train.csv', delimiter=',', header=0, na_values=[-1])  # 读取数据并将值为-1改为nan
    cross_rating = df.pivot_table(index='anime_id', columns='user_id', values='rating')  # 变成交叉表的格式 行为动漫，列为用户
    rating = cross_rating.values
    # relation表示某用户是否看过某部动漫（或者是否评分）看过为1，没看过为0
    relation = copy.deepcopy(rating)
    relation[np.where(np.isnan(relation))] = 0
    relation[np.where(relation != 0)] = 1
    return cross_rating, rating, relation


def load_test():
    df = pd.read_csv('rating_test.csv', delimiter=',', header=0, na_values=[-1])  # 读取数据并将值为-1改为nan
    df = df.values
    return df


class CF:
    def __init__(self, rating, relation, features, learning_rate=0.1, epochs=100):
        self.rating = rating
        self.relation = relation
        self.features = features  # 动漫和用户的特征(如：动作，爱情，冒险，校园，....)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.rows, self.columns = rating.shape
        self.loss_arr = []
        # 初始化w,x,d
        self.w = np.random.randn(self.rows, features)
        self.x = np.random.randn(self.columns, features)
        self.d = np.random.randn(self.rows)

    def grad(self):
        grad_w = np.zeros([self.rows, self.features])
        grad_x = np.zeros([self.columns, self.features])
        grad_d = np.zeros(self.rows)
        # 计算w,d的梯度
        for i in range(self.rows):
            index = np.where(self.relation[i, :] == 1)[0]  # 找出第i个动漫有人评分的下标
            grad_w[i] = np.dot(np.sum(self.w[i] * self.x[index], axis=1) + self.d[i] - self.rating[i, index],
                               self.x[index]) / len(index)
            grad_d[i] = np.sum(np.sum(self.w[i] * self.x[index], axis=1) + self.d[i] - self.rating[i, index]) / len(index)
        # 计算x的梯度
        for i in range(self.columns):
            index = np.where(self.relation[:, i] == 1)[0]  # 找出第i个人评分了的动漫的下标
            grad_x[i] = np.dot(np.sum(self.x[i] * self.w[index], axis=1) + self.d[index] - self.rating[index, i],
                               self.w[index]) / len(index)
        return grad_x, grad_w, grad_d

    def loss(self):
        loss_value = 0
        m, n = np.where(self.relation == 1)
        for i in range(len(m)):
            loss_value += np.square(np.sum(self.w[m[i]] * self.x[n[i]]) + self.d[m[i]] - self.rating[m[i], n[i]])
        return loss_value / len(m)

    def train(self):
        for epoch in range(self.epochs):
            grad_x, grad_w, grad_d = self.grad()
            # 梯度下降
            self.w = self.w - self.learning_rate * grad_w
            self.d = self.d - self.learning_rate * grad_d
            self.x = self.x - self.learning_rate * grad_x
            loss_value = self.loss()
            self.loss_arr.append(loss_value)
            print("第%d次迭代的loss:%f" % (epoch + 1, loss_value))
        return self.w, self.d, self.x

    def loss_graph(self):
        x = [i for i in range(self.epochs)]
        plt.figure(figsize=(6, 4))
        plt.plot(x, self.loss_arr, linewidth=1)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss of CollaborativeFiltering")
        plt.savefig('loss_CF.png', dpi=120, bbox_inches='tight')
        plt.show()


def recommend():
    """
    根据用户id和动漫id找到训练好的w,x,d计算评分
    比较真实的评分和预测的评分，计算召回率
    """
    length = len(test_data)
    true_positive_num = 0
    true_negative_num = 0
    false_positive_num = 0
    false_negative_num = 0
    for i in range(length):
        try:
            user_index = cross_rating.columns.get_loc(test_data[i, 0])
            anime_index = cross_rating.index.get_loc(test_data[i, 1])
        except:
            print("数据库中没有该动漫或用户的feature数据")
            continue
        score = np.sum(w[anime_index] * x[user_index]) + d[anime_index]
        print("第%d个-->预测的评分:%f,真实的评分:%f" % (i, score, test_data[i, 2]))
        # 评分在0~10之间，此处认为评分>=6的就推荐给用户
        if score >= 6 and test_data[i, 2] >= 6:
            true_positive_num += 1
        elif score < 6 and test_data[i, 2] < 6:
            true_negative_num += 1
        elif score >= 6 and test_data[i, 2] < 6:
            false_positive_num += 1
        elif score < 6 and test_data[i, 2] >= 6:
            false_negative_num += 1
    print("true_positive_num:%d" % true_positive_num)
    print("false_negative_num:%d" % false_negative_num)
    print("召回率:{:.2%}".format(true_positive_num / (true_positive_num + false_negative_num)))


if __name__ == "__main__":
    """
    训练数据和测试数据为下载得到rating.csv分割出来的一部分
    """
    cross_rating, rating, relation = load_train()  # 加载训练数据
    test_data = load_test()  # 加载测试数据
    model = CF(rating, relation, features=10, learning_rate=0.1, epochs=100)  # 创建模型
    w, d, x = model.train()  # 训练模型
    model.loss_graph()  # 画出损失图像
    recommend()  # 计算召回率

