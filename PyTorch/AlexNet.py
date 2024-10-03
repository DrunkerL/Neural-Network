import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np


# 定义网络模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2),  # 改为kernel_size=5
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 修改数据加载器以进行图像填充和调整
def data_loader(is_train):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # Resize to 224x224
        torchvision.transforms.ToTensor()
    ])
    data_set = torchvision.datasets.CIFAR10(
        root='./data', train=is_train, download=True,
        transform=transform
    )
    data_loader = DataLoader(data_set, batch_size=128, shuffle=is_train)
    return data_loader


# 训练模型（与之前相同）
def train(model, data_loader, loss_fn, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for data in data_loader:
            inputs, labels = data
            # 将数据移动到设备上
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


# 测试并可视化结果
def test_and_visualize(model, data_loader, class_names):
    model.eval()
    with torch.no_grad():
        data_iter = iter(data_loader)
        images, labels = next(data_iter)

        # 将数据移动到设备上
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # 可视化结果
        fig = plt.figure(figsize=(12, 8))
        for i in range(12):
            ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
            img = images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为 HWC 格式，并移动到 CPU
            img = np.clip(img, 0, 1)  # 限制在 [0, 1] 范围内
            ax.imshow(img)
            ax.set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[predicted[i]]}')
        plt.show()


# 主程序
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数和优化器
    model = AlexNet(num_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载训练和测试数据
    train_loader = data_loader(is_train=True)
    test_loader = data_loader(is_train=False)

    # 训练模型
    train(model, train_loader, loss_fn, optimizer, epochs=10)

    # 可视化测试结果
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    test_and_visualize(model, test_loader, class_names)

