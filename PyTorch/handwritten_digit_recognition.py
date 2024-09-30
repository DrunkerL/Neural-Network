import torch
import torchvision.datasets
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Linear, Softmax, ReLU
from torch.utils.data import DataLoader

# 定义训练网络使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            Linear(28 * 28, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, 10),
            Softmax(dim=1),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # 将数据展平
        return self.model(x)

# 加载数据
def get_data_loader(is_train):
    data_set = torchvision.datasets.MNIST(root='./data', train=is_train, download=True,
                                          transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(data_set, batch_size=16, shuffle=is_train)
    return dataloader

# 评估网络模型
def evaluate(model, data_loader):
    total_correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for img, label in data_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            total += label.size(0)  # 当前批次的样本数量
            total_correct += (output.argmax(dim=1) == label).sum().item()  # 统计正确预测的数量
    accuracy = total_correct / total
    return accuracy

# 训练网络模型
def train(model, data_loader, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for img, label in data_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
        # 每个epoch结束后进行评估
        train_accuracy = evaluate(model, data_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}')

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
    train_data = get_data_loader(True)
    test_data = get_data_loader(False)

    # 初始化模型，损失函数，优化器
    model = Net().to(device)
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
