# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler


# file_path = './data/USA.xlsx'

# df = pd.read_csv(file_path, usecols=[
#     '收盘价', '股指','近12月波动率(%)'
# ])

# # Try to read the file without using usecols first to check column names
# # df = pd.read_csv(file_path)

# # Print the column names to verify them
# print(df.columns.tolist())

# # 展示前几行以确认数据加载正确
# print(df.head())


# 由于代码执行环境被重置，需要重新导入必要的库并定义变量和函数

# 重新导入库
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler



def prepare_dataset(file_path_csv='./data/USA.csv'):
    df_csv = pd.read_csv(file_path_csv)

    # 准备输入（X）和标签（y）
    X_csv = df_csv.iloc[:, :2].values  # 选择前两列作为输入
    y_csv = df_csv.iloc[:, 2].values  # 选择第三列作为ylabel

    # 标准化特征
    scaler_csv = StandardScaler()
    X_csv_scaled = scaler_csv.fit_transform(X_csv)

    # 转换为tensor
    X_csv_tensor = torch.tensor(X_csv_scaled, dtype=torch.float32)
    y_csv_tensor = torch.tensor(y_csv, dtype=torch.float32).view(-1, 1)

    # 分割数据集
    X_train_csv_tensor = X_csv_tensor[140:240]
    y_train_csv_tensor = y_csv_tensor[140:240]
    X_test_csv_tensor = X_csv_tensor[240:]
    y_test_csv_tensor = y_csv_tensor[240:]

    # 显示数据集的tensor维度
    # print("Training set X shape:", X_train_csv_tensor.shape)
    # print("Training set y shape:", y_train_csv_tensor.shape)
    # print("Test set X shape:", X_test_csv_tensor.shape)
    # print("Test set y shape:", y_test_csv_tensor.shape)

    return [X_train_csv_tensor, y_train_csv_tensor, X_test_csv_tensor, y_test_csv_tensor]

prepare_dataset()