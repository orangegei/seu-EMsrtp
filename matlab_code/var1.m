% 假设Y是一个时间序列矩阵，其中每一列代表一个变量
Y = randn(100,2); % 这里我们使用随机数据作为示例

% 创建VAR模型的实例，指定滞后阶数为1
Model = varm('NumSeries', 2, 'NumLags', 1);

% 估计VAR模型
EstModel = estimate(Model, Y);

% 查看估计得到的系数
disp(EstModel.AR{:})

% 进行预测
[YForecast,~] = forecast(EstModel, 10, Y);
disp(YForecast)
