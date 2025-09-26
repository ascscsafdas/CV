#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLAXIS 3D 循环加载脚本测试
PLAXIS 3D Cyclic Loading Script Test

测试改进版PLAXIS 3D循环加载脚本的功能
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# 添加项目路径到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plaxis_cyclic_loading import (
    PlaxisCyclicLoader, 
    PlaxisConnectionError, 
    PlaxisLoadError, 
    PlaxisValidationError
)
from plaxis_utils import (
    check_system_resources,
    format_time_duration,
    format_file_size,
    calculate_progress_eta,
    PerformanceMonitor
)


class TestPlaxisCyclicLoader(unittest.TestCase):
    """测试PLAXIS循环加载器"""
    
    def setUp(self):
        """测试前设置"""
        self.loader = PlaxisCyclicLoader(
            load_value=100.0,
            total_cycles=10,  # 使用较小的数值进行测试
            batch_size=5,
            log_level='WARNING'  # 减少测试时的日志输出
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.loader.load_value, 100.0)
        self.assertEqual(self.loader.total_cycles, 10)
        self.assertEqual(self.loader.batch_size, 5)
        self.assertIsNotNone(self.loader.logger)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试无效的荷载值
        with self.assertRaises(PlaxisValidationError):
            PlaxisCyclicLoader(load_value=-10, total_cycles=10)
        
        # 测试无效的循环次数
        with self.assertRaises(PlaxisValidationError):
            PlaxisCyclicLoader(load_value=100, total_cycles=0)
        
        # 测试无效的批处理大小
        with self.assertRaises(PlaxisValidationError):
            PlaxisCyclicLoader(load_value=100, total_cycles=10, batch_size=-1)
    
    def test_batch_size_adjustment(self):
        """测试批处理大小自动调整"""
        loader = PlaxisCyclicLoader(
            load_value=100.0,
            total_cycles=5,
            batch_size=10,  # 大于总循环次数
            log_level='WARNING'
        )
        self.assertEqual(loader.batch_size, 5)  # 应该被调整为总循环次数
    
    def test_create_stage_load_calculation(self):
        """测试阶段创建时的荷载计算"""
        # 测试奇数阶段（加载）
        with patch.object(self.loader, '_activate_elements', return_value=True):
            with patch.object(self.loader, '_apply_load', return_value=True):
                # 模拟g_i对象
                mock_g_i = MagicMock()
                mock_g_i.create_stage = MagicMock()
                
                result = self.loader._create_stage(mock_g_i, 1)  # 奇数阶段
                self.assertTrue(result)
        
        # 测试偶数阶段（卸载）
        with patch.object(self.loader, '_activate_elements', return_value=True):
            with patch.object(self.loader, '_apply_load', return_value=True) as mock_apply:
                mock_g_i = MagicMock()
                mock_g_i.create_stage = MagicMock()
                
                result = self.loader._create_stage(mock_g_i, 2)  # 偶数阶段
                self.assertTrue(result)
                # 验证卸载时荷载值为0
                mock_apply.assert_called_with(mock_g_i, "Stage_2", 0.0)
    
    def test_error_handling(self):
        """测试错误处理"""
        with patch.object(self.loader, '_activate_elements', side_effect=Exception("测试错误")):
            mock_g_i = MagicMock()
            mock_g_i.create_stage = MagicMock()
            
            result = self.loader._create_stage(mock_g_i, 1)
            self.assertFalse(result)
            self.assertEqual(self.loader.stats['failed_stages'], 1)
    
    def test_stats_tracking(self):
        """测试统计信息跟踪"""
        self.assertEqual(self.loader.stats['completed_stages'], 0)
        self.assertEqual(self.loader.stats['failed_stages'], 0)
        self.assertEqual(len(self.loader.stats['errors']), 0)


class TestPlaxisUtils(unittest.TestCase):
    """测试工具函数"""
    
    def test_format_time_duration(self):
        """测试时间格式化"""
        self.assertIn("秒", format_time_duration(30))
        self.assertIn("分钟", format_time_duration(120))
        self.assertIn("小时", format_time_duration(3600))
    
    def test_format_file_size(self):
        """测试文件大小格式化"""
        self.assertIn("B", format_file_size(500))
        self.assertIn("KB", format_file_size(1500))
        self.assertIn("MB", format_file_size(1500000))
    
    def test_check_system_resources(self):
        """测试系统资源检查"""
        resources = check_system_resources()
        if 'error' not in resources:
            self.assertIn('memory', resources)
            self.assertIn('cpu', resources)
            self.assertIn('disk', resources)
    
    def test_calculate_progress_eta(self):
        """测试进度计算"""
        import time
        start_time = time.time() - 10  # 假设10秒前开始
        progress = calculate_progress_eta(5, 10, start_time)
        
        self.assertEqual(progress['progress_percent'], 50.0)
        self.assertIsNotNone(progress['eta_seconds'])
        self.assertIsNotNone(progress['eta_formatted'])
    
    def test_performance_monitor(self):
        """测试性能监控器"""
        monitor = PerformanceMonitor()
        
        import time
        time.sleep(0.1)  # 等待一小段时间
        
        monitor.add_checkpoint('test')
        elapsed = monitor.get_elapsed_time()
        
        self.assertGreater(elapsed, 0.05)  # 应该大于50ms
        
        checkpoint_elapsed = monitor.get_elapsed_time('test')
        self.assertLess(checkpoint_elapsed, elapsed)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_small_scale_run(self):
        """测试小规模运行"""
        loader = PlaxisCyclicLoader(
            load_value=50.0,
            total_cycles=4,  # 很小的循环次数
            batch_size=2,
            log_level='ERROR'  # 只显示错误
        )
        
        # 运行循环加载
        results = loader.run_cyclic_loading()
        
        # 验证结果
        self.assertIsInstance(results, dict)
        self.assertIn('completed_stages', results)
        self.assertIn('failed_stages', results)
        self.assertIn('start_time', results)
        self.assertIn('end_time', results)
        
        # 在模拟环境下，所有阶段都应该成功
        self.assertEqual(results['completed_stages'], 4)
        self.assertEqual(results['failed_stages'], 0)


def run_performance_test():
    """运行性能测试"""
    print("运行性能测试...")
    
    loader = PlaxisCyclicLoader(
        load_value=100.0,
        total_cycles=100,  # 中等规模测试
        batch_size=20,
        log_level='WARNING'
    )
    
    monitor = PerformanceMonitor()
    results = loader.run_cyclic_loading()
    total_time = monitor.get_elapsed_time()
    
    print(f"性能测试结果:")
    print(f"  总循环次数: {loader.total_cycles}")
    print(f"  成功完成: {results['completed_stages']}")
    print(f"  总耗时: {format_time_duration(total_time)}")
    print(f"  平均每阶段: {total_time/results['completed_stages']:.3f} 秒")
    
    # 检查系统资源
    resources = check_system_resources()
    if 'error' not in resources:
        print(f"  内存使用: {resources['memory']['percent_used']:.1f}%")
        print(f"  CPU使用: {resources['cpu']['percent_used']:.1f}%")


def main():
    """主测试函数"""
    print("PLAXIS 3D 循环加载脚本测试")
    print("=" * 50)
    
    # 运行单元测试
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 50)
    
    # 运行性能测试
    run_performance_test()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()