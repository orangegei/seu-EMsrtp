import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        # 定义一个卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        # 应用卷积层
        x = F.relu(self.conv1(x))
        # 应用池化层
        x = self.pool(x)
        return x


class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))
        
        return out

class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        self.cnn = CNNModule()
        self.lstm = LSTMModule(input_size=16*14*14, hidden_size=128, num_layers=2)
        self.fc = nn.Linear(128, 10)  # 假设我们的任务是10类分类
        
    def forward(self, x):
        # 假设x的形状是(batch_size, timesteps, C, H, W)
        batch_size, timesteps, C, H, W = x.size()
        # 将x调整为(batch_size * timesteps, C, H, W)以输入CNN
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        # 将CNN的输出调整为(batch_size, timesteps, -1)以输入LSTM
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out = self.lstm(r_in)
        # 只取LSTM最后一个时间步的输出进行分类
        r_out = r_out[:, -1, :]
        output = self.fc(r_out)
        
        return output
