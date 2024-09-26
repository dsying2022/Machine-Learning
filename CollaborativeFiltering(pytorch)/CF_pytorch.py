import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


def load_train():
    """导入训练数据"""
    df = pd.read_csv('rating_train.csv', delimiter=',', header=0, na_values=[-1])  # 读取数据并将值为-1改为nan
    cross_rating = df.pivot_table(index='anime_id', columns='user_id', values='rating')  # 变成交叉表的格式 行为动漫，列为用户
    rating = cross_rating.values
    rating[np.where(np.isnan(rating))] = 0
    # relation表示某用户是否看过某部动漫（或者是否评分）看过为1，没看过为0
    relation = copy.deepcopy(rating)
    relation[np.where(np.isnan(relation))] = 0
    relation[np.where(relation != 0)] = 1
    return cross_rating, rating, relation


def load_test():
    """导入测试数据"""
    df = pd.read_csv('rating_test.csv', delimiter=',', header=0, na_values=[-1])  # 读取数据并将值为-1改为nan
    df = df.values
    return df


def loss_graph(epochs, loss_arr):
    """画loss曲线"""
    x = [i for i in range(epochs)]
    plt.figure(figsize=(6, 4))
    plt.plot(x, loss_arr, linewidth=1)
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
            user_index = cross_rating.columns.get_loc(test_data[i, 0])  # 找到该用户id在矩阵中的下标
            anime_index = cross_rating.index.get_loc(test_data[i, 1])  # 找到该动漫id在矩阵中的下标
        except:
            print("数据库中没有该动漫或用户的feature数据")
            continue
        score = np.sum(w[anime_index] * x[user_index]) + b[anime_index, 0]  # 计算预测的评分
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


class CF(nn.Module):
    def __init__(self, relation, features):
        super(CF, self).__init__()
        self.rows, self.columns = relation.shape
        # 初始化w,x,d
        self.w = nn.Parameter(torch.randn(self.rows, features), requires_grad=True)
        self.x = nn.Parameter(torch.randn(self.columns, features), requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.rows, 1), requires_grad=True)

    def forward(self):
        return torch.matmul(self.w, self.x.T) + self.b


def train(rating, relation, features=10, epochs=100, learning_rate=0.1):
    rating = torch.tensor(rating, requires_grad=False)
    relation = torch.tensor(relation, requires_grad=False)
    # 设置损失函数
    loss_func = F.mse_loss
    model = CF(relation, features)
    # 设置优化器
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 开始训练
    loss_arr = []
    for epoch in range(epochs):
        loss = loss_func(model() * relation, rating)
        loss.backward()
        opt.step()
        opt.zero_grad()
        loss_arr.append(loss.data)
        print("loss:", loss.data)
    loss_graph(epochs, loss_arr)
    para = model.state_dict()
    w = para['w'].numpy()
    b = para['b'].numpy()
    x = para['x'].numpy()
    return w, b, x


if __name__ == "__main__":
    # 加载数据
    cross_rating, rating, relation = load_train()
    test_data = load_test()
    # 训练模型
    w, b, x = train(rating, relation, features=10, epochs=100, learning_rate=0.3)
    # 模型评估
    recommend()
