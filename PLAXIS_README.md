# PLAXIS 3D 循环加载脚本 - 改进版本

## 概述

这是一个改进版的PLAXIS 3D循环加载脚本，用于执行循环加载分析。脚本支持1000个计算阶段的循环加载，其中奇数阶段进行加载(Qx=100)，偶数阶段进行卸载(Qx=0)。

## 主要改进

### 1. 完善的错误处理
- 🛡️ PLAXIS连接失败处理
- 🔍 点荷载不存在检测
- 🔄 自动重试机制
- 📊 错误统计和报告

### 2. 优化的代码结构
- 🏗️ 面向对象设计
- 🔧 公共函数提取
- 📦 模块化架构
- 🎯 单一职责原则

### 3. 参数验证
- ✅ 输入参数有效性检查
- 🔢 数值范围验证
- 🚨 早期错误发现
- 📝 详细错误信息

### 4. 详细日志系统
- 📋 多级别日志记录
- 📈 执行进度跟踪
- 💾 日志文件保存
- 🖥️ 控制台输出

### 5. 性能优化
- 🚀 批量处理机制
- 🧹 内存管理优化
- ⚡ 垃圾回收策略
- 📊 性能监控

### 6. 系统资源监控
- 💾 内存使用监控
- 🖥️ CPU使用监控
- 💽 磁盘空间检查
- ⚠️ 资源预警

## 文件结构

```
├── plaxis_cyclic_loading.py    # 主脚本文件
├── plaxis_config.py           # 配置文件
├── plaxis_utils.py            # 工具函数模块
├── test_plaxis_cyclic.py      # 测试脚本
└── PLAXIS_README.md           # 说明文档
```

## 安装要求

### Python 依赖
```bash
pip install psutil
```

### PLAXIS 依赖
- PLAXIS 3D (版本 2018 或更新)
- plxscripting 模块

## 使用方法

### 基本使用

```python
from plaxis_cyclic_loading import PlaxisCyclicLoader

# 创建加载器实例
loader = PlaxisCyclicLoader(
    load_value=100.0,      # 荷载值
    total_cycles=1000,     # 总循环次数
    batch_size=50,         # 批处理大小
    log_level='INFO'       # 日志级别
)

# 执行循环加载
results = loader.run_cyclic_loading()

# 查看结果
print(f"成功完成: {results['completed_stages']} 阶段")
print(f"失败次数: {results['failed_stages']} 阶段")
```

### 高级配置

```python
# 使用配置文件
from plaxis_config import LOADING_CONFIG, LOGGING_CONFIG

loader = PlaxisCyclicLoader(
    load_value=LOADING_CONFIG['load_value'],
    total_cycles=LOADING_CONFIG['total_cycles'],
    batch_size=LOADING_CONFIG['batch_size'],
    log_level=LOGGING_CONFIG['level']
)
```

### 命令行使用

```bash
python plaxis_cyclic_loading.py
```

## 配置选项

### 加载配置 (plaxis_config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `load_value` | 100.0 | 加载时的荷载值 |
| `unload_value` | 0.0 | 卸载时的荷载值 |
| `total_cycles` | 1000 | 总循环次数 |
| `batch_size` | 50 | 批处理大小 |
| `load_direction` | 'Qx' | 荷载方向 |

### 日志配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `level` | 'INFO' | 日志级别 |
| `file_name` | 'plaxis_cyclic_loading.log' | 日志文件名 |
| `console_output` | True | 控制台输出 |

### 性能配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `memory_threshold` | 85 | 内存使用阈值 (%) |
| `gc_frequency` | 10 | 垃圾回收频率 |
| `progress_report_interval` | 10 | 进度报告间隔 |

## 错误处理

### 异常类型

- `PlaxisConnectionError`: PLAXIS连接失败
- `PlaxisLoadError`: 荷载应用失败  
- `PlaxisValidationError`: 参数验证失败

### 错误恢复策略

1. **连接错误**: 自动重试连接
2. **荷载错误**: 跳过当前阶段，继续执行
3. **验证错误**: 立即停止执行
4. **系统资源不足**: 等待资源释放后继续

## 性能优化建议

### 内存优化
- 使用较小的批处理大小 (20-50)
- 定期执行垃圾回收
- 监控内存使用情况

### 计算优化
- 根据系统性能调整批处理大小
- 使用SSD存储提高I/O性能
- 确保足够的可用内存

### 网络优化
- 在本地运行PLAXIS服务器
- 使用稳定的网络连接
- 避免防火墙干扰

## 测试

### 运行单元测试
```bash
python test_plaxis_cyclic.py
```

### 运行性能测试
```bash
python -c "from test_plaxis_cyclic import run_performance_test; run_performance_test()"
```

## 监控和调试

### 日志级别
- `DEBUG`: 详细调试信息
- `INFO`: 一般信息 (推荐)
- `WARNING`: 警告信息
- `ERROR`: 仅错误信息

### 性能监控
- 实时内存使用监控
- CPU使用率跟踪
- 执行时间统计
- 进度预估

### 调试技巧
1. 使用小规模测试 (10-20个循环)
2. 启用DEBUG日志级别
3. 监控系统资源使用
4. 检查PLAXIS连接稳定性

## 常见问题

### Q: 脚本运行缓慢怎么办？
A: 
- 减小批处理大小
- 检查系统资源使用
- 优化PLAXIS模型复杂度
- 使用更快的硬件

### Q: 连接PLAXIS失败怎么办？
A:
- 检查PLAXIS是否正常运行
- 验证端口配置 (默认: 10000, 10001)
- 检查防火墙设置
- 确认密码配置

### Q: 内存不足怎么办？
A:
- 减小批处理大小
- 关闭不必要的程序
- 增加虚拟内存
- 使用64位系统

### Q: 如何自定义荷载模式？
A:
- 修改`_create_stage`方法中的荷载计算逻辑
- 调整`plaxis_config.py`中的配置参数
- 实现自定义荷载函数

## 许可证

此脚本仅供学习和研究使用。使用时请遵守PLAXIS软件的许可协议。

## 支持

如有问题或建议，请通过以下方式联系：
- 提交Issue到代码仓库
- 发送邮件到技术支持
- 查看PLAXIS官方文档

---

**注意**: 在生产环境中使用前，请务必进行充分测试，并备份重要数据。