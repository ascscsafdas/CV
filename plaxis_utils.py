#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLAXIS 3D 工具函数模块
PLAXIS 3D Utilities Module

包含PLAXIS 3D循环加载脚本使用的工具函数
"""

import os
import psutil
import time
import json
from typing import Dict, Any, Optional
from functools import wraps


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试延迟（秒）
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        continue
                    else:
                        raise last_exception
            return None
        return wrapper
    return decorator


def timing_decorator(func):
    """
    计时装饰器
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.3f} 秒")
        return result
    return wrapper


def check_system_resources() -> Dict[str, Any]:
    """
    检查系统资源
    
    Returns:
        Dict[str, Any]: 系统资源信息
    """
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('.')
        
        return {
            'memory': {
                'total': memory.total / (1024**3),  # GB
                'available': memory.available / (1024**3),  # GB
                'percent_used': memory.percent
            },
            'cpu': {
                'percent_used': cpu_percent
            },
            'disk': {
                'total': disk_usage.total / (1024**3),  # GB
                'free': disk_usage.free / (1024**3),  # GB
                'percent_used': (disk_usage.used / disk_usage.total) * 100
            }
        }
    except Exception as e:
        return {'error': f'无法获取系统资源信息: {str(e)}'}


def validate_file_path(file_path: str) -> bool:
    """
    验证文件路径
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 路径有效返回True
    """
    try:
        return os.path.exists(os.path.dirname(file_path))
    except Exception:
        return False


def create_backup_file(file_path: str) -> Optional[str]:
    """
    创建备份文件
    
    Args:
        file_path: 原文件路径
        
    Returns:
        Optional[str]: 备份文件路径，失败返回None
    """
    try:
        if os.path.exists(file_path):
            timestamp = int(time.time())
            backup_path = f"{file_path}.backup.{timestamp}"
            import shutil
            shutil.copy2(file_path, backup_path)
            return backup_path
    except Exception:
        pass
    return None


def format_time_duration(seconds: float) -> str:
    """
    格式化时间持续时间
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} 分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} 小时"


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        str: 格式化的文件大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def save_results_to_json(results: Dict[str, Any], file_path: str) -> bool:
    """
    保存结果到JSON文件
    
    Args:
        results: 结果字典
        file_path: 文件路径
        
    Returns:
        bool: 保存成功返回True
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 转换不可序列化的对象
        serializable_results = make_serializable(results)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception:
        return False


def make_serializable(obj: Any) -> Any:
    """
    将对象转换为可JSON序列化的格式
    
    Args:
        obj: 要转换的对象
        
    Returns:
        Any: 可序列化的对象
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def calculate_progress_eta(completed: int, total: int, start_time: float) -> Dict[str, Any]:
    """
    计算进度和预计完成时间
    
    Args:
        completed: 已完成数量
        total: 总数量
        start_time: 开始时间戳
        
    Returns:
        Dict[str, Any]: 进度信息
    """
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if completed == 0:
        return {
            'progress_percent': 0.0,
            'elapsed_time': elapsed_time,
            'eta_seconds': None,
            'eta_formatted': "未知"
        }
    
    progress_percent = (completed / total) * 100
    avg_time_per_item = elapsed_time / completed
    remaining_items = total - completed
    eta_seconds = remaining_items * avg_time_per_item
    
    return {
        'progress_percent': progress_percent,
        'elapsed_time': elapsed_time,
        'eta_seconds': eta_seconds,
        'eta_formatted': format_time_duration(eta_seconds)
    }


def print_progress_bar(completed: int, total: int, width: int = 50, prefix: str = "进度"):
    """
    打印进度条
    
    Args:
        completed: 已完成数量
        total: 总数量
        width: 进度条宽度
        prefix: 前缀文本
    """
    progress = completed / total
    filled_width = int(width * progress)
    bar = '█' * filled_width + '░' * (width - filled_width)
    percent = progress * 100
    
    print(f'\r{prefix}: |{bar}| {percent:.1f}% ({completed}/{total})', end='', flush=True)
    
    if completed == total:
        print()  # 完成时换行


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        
    def add_checkpoint(self, name: str) -> None:
        """添加检查点"""
        self.checkpoints[name] = time.time()
    
    def get_elapsed_time(self, checkpoint: str = None) -> float:
        """获取经过的时间"""
        if checkpoint and checkpoint in self.checkpoints:
            return time.time() - self.checkpoints[checkpoint]
        return time.time() - self.start_time
    
    def get_checkpoint_duration(self, start_checkpoint: str, end_checkpoint: str) -> float:
        """获取两个检查点之间的时间"""
        if start_checkpoint in self.checkpoints and end_checkpoint in self.checkpoints:
            return self.checkpoints[end_checkpoint] - self.checkpoints[start_checkpoint]
        return 0.0
    
    def reset(self) -> None:
        """重置计时器"""
        self.start_time = time.time()
        self.checkpoints.clear()


def cleanup_temp_files(directory: str, pattern: str = "*.tmp") -> int:
    """
    清理临时文件
    
    Args:
        directory: 目录路径
        pattern: 文件模式
        
    Returns:
        int: 清理的文件数量
    """
    import glob
    
    cleaned_count = 0
    try:
        temp_files = glob.glob(os.path.join(directory, pattern))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                cleaned_count += 1
            except Exception:
                continue
    except Exception:
        pass
    
    return cleaned_count