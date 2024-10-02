import torch
import torchvision.datasets
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

#定义训练样本使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#定义LeNet-5网络结构
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,6,5,1,0),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Conv2d(6,16,5,1,0),
            nn.ReLU(),
            nn.AvgPool2d(2,2),
            nn.Flatten(),
            nn.Linear(16*4*4,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)

#加载数据
def load_data(is_train):
    data_set = torchvision.datasets.MNIST(root='./data', train=is_train, download=True,
                                          transform=torchvision.transforms.ToTensor())
    data_loader = DataLoader(data_set, batch_size=64, shuffle=is_train)
    return data_loader


#评估网络模型
def evaluate(model,data_loader):
    correct = 0  #预测正确样本的数量
    total = 0    #当前批次样本的总量
    with torch.no_grad():
        model.eval()
        for data in data_loader:
            img, label = data
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            output = model(img)
            total += label.size(0)  # 当前批次的样本数量
            correct += (output.argmax(dim=1) == label).sum().item()  # 统计正确预测的数量
        accuracy = 100 * correct / total
        return accuracy

#训练模型
def train(model, data_loader, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for data in data_loader:
            img, label = data
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            output = model(img)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 每个epoch结束后进行评估
        train_accuracy = evaluate(model, data_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}')


#可视化测试结果
def visualize_predictions(model, data_loader):
    model.eval()
    imgs, labels = next(iter(data_loader))  # 获取一个批次的数据
    imgs = imgs.to(device)  # 确保输入在同一设备上
    preds = model(imgs).argmax(dim=1).cpu().numpy()  # 预测
    labels = labels.cpu().numpy()

    # 设置图形参数
    plt.figure(figsize=(12, 6))
    for i in range(16):  # 显示前16个样本
        plt.subplot(4, 4, i + 1)
        plt.imshow(imgs[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Pred: {preds[i]}, True: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # 准备数据集
    train_data = load_data(True)
    test_data = load_data(False)

    # 初始化模型，损失函数，优化器
    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_data, loss_fn, optimizer, epochs=10)

    # 在测试集上评估模型
    test_accuracy = evaluate(model, test_data)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # 可视化预测结果
    visualize_predictions(model, test_data)

if __name__ == '__main__':
    main()
