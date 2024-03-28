import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

file_path = './data/Financial_Stress_Index_1.csv'

# 步骤1: 读取数据
df = pd.read_csv(file_path, usecols=[
    'Date', 'Japan FSI', 'Japan return', 'Japan Equity volatility',
    'Japan Bank beta', 'Japan Debt spread', 'Japan EMPI'
])

# 步骤2: 预处理数据
# 使用'Japan FSI'列作为目标（target）
targets = df[['Japan FSI']]

# 使用其他列作为特征（features）
features = df[['Japan return', 'Japan Equity volatility', 'Japan Bank beta', 'Japan Debt spread', 'Japan EMPI']]

# 规范化数据
scaler = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler.fit_transform(features)
# targets_scaled = scaler.fit_transform(targets)

# 转换为PyTorch张量
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

features_tensor_unsqueezed = features_tensor.unsqueeze(0)
targets_tensor_unsqueezed = targets_tensor.unsqueeze(0)

# 步骤3: 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_tensor, targets_tensor, test_size=0.2, random_state=42)

# 创建数据加载器
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64  # 可以根据需要调整
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # 选择LSTM最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


sample_input = torch.rand(1, 10, 5)  # (批量大小, 序列长度, 特征数量)
# 实例化模型、定义损失函数和优化器
model = LSTMModel(input_dim=5, hidden_dim=20, output_dim=1, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs) # inputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 在测试集上评估模型
with torch.no_grad():
    total_loss = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader)}')
