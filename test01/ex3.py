from getdata import *
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim

# # 重新加载和准备数据
# file_path = '/mnt/data/USA.xlsx'
# df = pd.read_excel(file_path)

# X = df[['收盘价', '股指']].values
# y = df['近12月波动率(%)'].values

# # 标准化特征
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 转换为tensor
# X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# # 分割数据集
# X_train_tensor = X_tensor[:200]
# y_train_tensor = y_tensor[:200]
# X_test_tensor = X_tensor[200:]
# y_test_tensor = y_tensor[200:]

X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_dataset()

# 创建PyTorch数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义一个简单的线性模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入特征维度为2，输出为1

    def forward(self, x):
        return self.linear(x)

# 实例化模型、损失函数和优化器
model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')
