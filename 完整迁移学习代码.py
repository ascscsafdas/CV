#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整迁移学习代码 - 多模型对比与预测
包含多个深度学习模型的迁移学习实现
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class SeqGTAN(nn.Module):
    """Seq-GTAN模型定义"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2):
        super(SeqGTAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) 
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # 输入投影
        x = self.input_projection(x)
        
        # 多层注意力机制
        for i in range(self.num_layers):
            residual = x
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms[i](residual + self.dropout(attn_output))
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        output = self.output_layer(lstm_out[:, -1, :])
        
        return output

class SimpleRNN(nn.Module):
    """简单RNN模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    """简单LSTM模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class SimpleGRU(nn.Module):
    """简单GRU模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class SimpleTransformer(nn.Module):
    """简单Transformer模型"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(SimpleTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out

def create_sequences(data, seq_length):
    """创建时间序列数据"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def transfer_learning_south_mine():
    """南矿迁移学习主函数"""
    print("开始南矿迁移学习实验...")
    
    # 生成模拟数据 (实际使用时替换为真实数据)
    np.random.seed(42)
    time_steps = 1000
    t = np.linspace(0, 10, time_steps)
    # 模拟南矿数据：包含趋势、季节性和噪声
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 365) + 5 * np.cos(4 * np.pi * t / 365)
    noise = np.random.normal(0, 2, time_steps)
    data = trend + seasonal + noise + 50  # 基础值50
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 创建序列数据
    seq_length = 30
    X, y = create_sequences(data_scaled, seq_length)
    
    # 划分数据集
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1)
    X_val = torch.FloatTensor(X_val).unsqueeze(-1)
    y_val = torch.FloatTensor(y_val).unsqueeze(-1)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 模型配置
    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    
    # 定义模型 - 只使用Seq-GTAN
    model_types = ['Seq-GTAN']
    models = {
        'Seq-GTAN': SeqGTAN(input_dim, hidden_dim, output_dim)
    }
    
    # 训练所有模型并收集结果
    results = {}
    predictions = {}
    
    for model_name in model_types:
        print(f"\n训练 {model_name} 模型...")
        model = models[model_name]
        
        # 训练模型
        train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=50)
        
        # 测试预测
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).numpy()
        
        # 反标准化
        y_test_original = scaler.inverse_transform(y_test.numpy())
        y_pred_original = scaler.inverse_transform(y_pred)
        
        # 计算评估指标
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        results[model_name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        predictions[model_name] = {
            'actual': y_test_original.flatten(),
            'predicted': y_pred_original.flatten()
        }
        
        print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # 绘制预测结果
    plot_predictions(predictions, model_types)
    
    # 保存结果到Excel
    save_results_to_excel(results, predictions, model_types)
    
    return results, predictions

def plot_predictions(predictions, model_types):
    """绘制预测结果对比图 - 只显示Seq-GTAN结果"""
    plt.figure(figsize=(12, 8))
    
    # 定义颜色 - 只需要两种颜色：真实值和Seq-GTAN预测值
    colors = ['black', '#9467bd']  # 黑色表示真实值，紫色表示Seq-GTAN
    
    # 绘制真实值
    actual_data = predictions['Seq-GTAN']['actual']
    plt.plot(actual_data[:200], color=colors[0], linewidth=2, label='真实值', alpha=0.8)
    
    # 绘制Seq-GTAN预测值
    predicted_data = predictions['Seq-GTAN']['predicted']
    plt.plot(predicted_data[:200], 
            color=colors[1], 
            linestyle='-', 
            linewidth=2, 
            label='Seq-GTAN预测值', 
            alpha=0.8)
    
    plt.title('南矿Seq-GTAN迁移学习预测结果', fontsize=16, fontweight='bold')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('南矿Seq-GTAN迁移学习预测结果.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_excel(results, predictions, model_types):
    """保存结果到Excel文件 - 只包含Seq-GTAN结果"""
    # 创建评估指标DataFrame - 只包含Seq-GTAN
    metrics_data = [{
        '模型': 'Seq-GTAN',
        'MSE': results['Seq-GTAN']['MSE'],
        'MAE': results['Seq-GTAN']['MAE'],
        'R2': results['Seq-GTAN']['R2']
    }]
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # 创建预测结果DataFrame - 只包含真实值和Seq-GTAN预测值
    pred_data = {
        '真实值': predictions['Seq-GTAN']['actual'],
        'Seq-GTAN预测值': predictions['Seq-GTAN']['predicted']
    }
    
    predictions_df = pd.DataFrame(pred_data)
    
    # 保存到Excel
    with pd.ExcelWriter('南矿Seq-GTAN迁移学习结果.xlsx', engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='评估指标', index=False)
        predictions_df.to_excel(writer, sheet_name='预测结果', index=False)
    
    print("结果已保存到 '南矿Seq-GTAN迁移学习结果.xlsx'")

if __name__ == "__main__":
    # 运行迁移学习实验
    results, predictions = transfer_learning_south_mine()
    print("\n南矿Seq-GTAN迁移学习实验完成！")