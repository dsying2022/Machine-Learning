import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def load_data():
    """加载mnist数据"""
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(root='../dataset/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


class Net(torch.nn.Module):
    """构建卷积神经网络模型"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(16*4*4, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


def train(train_loader, epochs):
    """训练模型"""
    loss_arr = []
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())

            print("第%d次迭代第%d批的loss:%f" % (epoch, i, loss.item()))
    loss_graph(loss_arr)


def test(test_loader):
    """测试模型"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("测试数据总个数:%d" % total)
    print("预测正确的数据个数%d" % correct)
    print("正确率:{:.2%}".format(correct / total))


def loss_graph(loss_arr):
    x = [i for i in range(len(loss_arr))]
    plt.figure(figsize=(6, 4))
    plt.plot(x, loss_arr, linewidth=1)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss")
    plt.savefig('loss_cnn.png', dpi=120, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    train_loader, test_loader = load_data()  # 加载数据
    model = Net()  # 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 定义使用的设备
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 计算loss
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 构建优化器
    # 训练模型
    train(train_loader, epochs=10)
    # 测试模型
    test(test_loader)