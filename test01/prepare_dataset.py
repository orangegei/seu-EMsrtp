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

print(features_tensor.shape)
print(targets_tensor.shape)

features_tensor_unsqueezed = features_tensor.unsqueeze(0)
print(features_tensor_unsqueezed.shape)

# 步骤3: 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_tensor, targets_tensor, test_size=0.2, random_state=42)

# 创建数据加载器
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64  # 可以根据需要调整
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

print(train_data)
print(test_data)