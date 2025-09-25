#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试Seq-GTAN迁移学习代码的基本功能
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 修改后的测试版本，减少训练轮数以快速验证
def test_seq_gtan():
    """测试Seq-GTAN模型的基本功能"""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from torch.utils.data import TensorDataset, DataLoader
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("开始测试Seq-GTAN模型...")
    
    # SeqGTAN模型定义（从原文件复制）
    class SeqGTAN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2):
            super(SeqGTAN, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) 
                for _ in range(num_layers)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = self.input_projection(x)
            for i in range(self.num_layers):
                residual = x
                attn_output, _ = self.attention_layers[i](x, x, x)
                x = self.layer_norms[i](residual + self.dropout(attn_output))
            lstm_out, _ = self.lstm(x)
            output = self.output_layer(lstm_out[:, -1, :])
            return output
    
    # 生成测试数据
    time_steps = 200  # 减少数据量以加快测试
    t = np.linspace(0, 10, time_steps)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.cos(4 * np.pi * t / 50)
    noise = np.random.normal(0, 2, time_steps)
    data = trend + seasonal + noise + 50
    
    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 创建序列数据
    def create_sequences(data, seq_length):
        sequences, targets = [], []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            target = data[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    seq_length = 10  # 减少序列长度
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
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    input_dim = 1
    hidden_dim = 32  # 减少隐藏层维度
    output_dim = 1
    model = SeqGTAN(input_dim, hidden_dim, output_dim, num_heads=4, num_layers=1)
    
    # 训练模型（快速测试版本）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练...")
    model.train()
    for epoch in range(5):  # 只训练5个epoch用于测试
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')
    
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
    
    print(f"\n测试结果:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    # 简单绘图测试
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original.flatten()[:50], label='真实值', color='black', linewidth=2)
    plt.plot(y_pred_original.flatten()[:50], label='Seq-GTAN预测值', color='purple', linewidth=2)
    plt.title('Seq-GTAN测试结果')
    plt.xlabel('时间步')
    plt.ylabel('预测值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('seq_gtan_test_result.png', dpi=150, bbox_inches='tight')
    print("测试图表已保存为 'seq_gtan_test_result.png'")
    
    print("\nSeq-GTAN模型测试完成！")
    return True

if __name__ == "__main__":
    try:
        success = test_seq_gtan()
        if success:
            print("✓ 所有测试通过")
        else:
            print("✗ 测试失败")
    except Exception as e:
        print(f"✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()