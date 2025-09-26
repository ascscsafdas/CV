#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLAXIS 3D 循环加载配置文件
PLAXIS 3D Cyclic Loading Configuration

此文件包含PLAXIS 3D循环加载脚本的所有配置参数
"""

# PLAXIS连接配置
PLAXIS_CONFIG = {
    'input_server': {
        'host': 'localhost',
        'port': 10000,
        'password': ''
    },
    'output_server': {
        'host': 'localhost', 
        'port': 10001,
        'password': ''
    },
    'timeout': 30  # 连接超时时间（秒）
}

# 循环加载配置
LOADING_CONFIG = {
    'load_value': 100.0,        # 加载时的荷载值
    'unload_value': 0.0,        # 卸载时的荷载值
    'total_cycles': 1000,       # 总循环次数
    'batch_size': 50,           # 批处理大小
    'load_direction': 'Qx',     # 荷载方向 ('Qx', 'Qy', 'Qz')
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',            # 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    'file_name': 'plaxis_cyclic_loading.log',
    'max_file_size': 10,        # 最大日志文件大小 (MB)
    'backup_count': 5,          # 保留的日志文件数量
    'console_output': True      # 是否输出到控制台
}

# 性能优化配置
PERFORMANCE_CONFIG = {
    'memory_threshold': 85,     # 内存使用阈值 (%)
    'gc_frequency': 10,         # 垃圾回收频率（每n个批次）
    'progress_report_interval': 10,  # 进度报告间隔（每n个阶段）
    'auto_save_interval': 100   # 自动保存间隔（每n个阶段）
}

# 错误处理配置
ERROR_CONFIG = {
    'max_retries': 3,           # 最大重试次数
    'retry_delay': 1.0,         # 重试延迟（秒）
    'continue_on_error': True,  # 遇到错误是否继续
    'fail_threshold': 0.1       # 失败率阈值（超过此比例将停止执行）
}

# 验证配置
VALIDATION_CONFIG = {
    'check_model_validity': True,    # 是否检查模型有效性
    'check_point_loads': True,       # 是否检查点荷载存在
    'check_available_memory': True,  # 是否检查可用内存
    'min_available_memory': 1024     # 最小可用内存 (MB)
}