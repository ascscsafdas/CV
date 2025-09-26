#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PLAXIS 3D 循环加载脚本 - 改进版本
PLAXIS 3D Cyclic Loading Script - Improved Version

功能：
- 奇数阶段加载 (Qx=100)
- 偶数阶段卸载 (Qx=0) 
- 循环1000次
- 完善的错误处理
- 优化的代码结构
- 参数验证
- 详细日志记录
- 批量处理优化

作者: PLAXIS Script Team
版本: 2.0
"""

import logging
import time
import sys
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class PlaxisConnectionError(Exception):
    """PLAXIS连接异常"""
    pass


class PlaxisLoadError(Exception):
    """PLAXIS荷载异常"""
    pass


class PlaxisValidationError(Exception):
    """PLAXIS参数验证异常"""
    pass


class PlaxisCyclicLoader:
    """PLAXIS 3D 循环加载器"""
    
    def __init__(self, 
                 load_value: float = 100.0,
                 total_cycles: int = 1000,
                 batch_size: int = 50,
                 log_level: str = 'INFO'):
        """
        初始化循环加载器
        
        Args:
            load_value: 荷载值 (默认: 100.0)
            total_cycles: 总循环次数 (默认: 1000)
            batch_size: 批处理大小 (默认: 50)
            log_level: 日志级别 (默认: 'INFO')
        """
        self.load_value = load_value
        self.total_cycles = total_cycles
        self.batch_size = batch_size
        self.g_i = None  # PLAXIS输入对象
        self.g_o = None  # PLAXIS输出对象
        
        # 设置日志
        self._setup_logging(log_level)
        
        # 验证参数
        self._validate_parameters()
        
        # 统计信息
        self.stats = {
            'completed_stages': 0,
            'failed_stages': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
    
    def _setup_logging(self, log_level: str) -> None:
        """设置日志系统"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler('plaxis_cyclic_loading.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("日志系统初始化完成")
    
    def _validate_parameters(self) -> None:
        """验证输入参数"""
        if not isinstance(self.load_value, (int, float)) or self.load_value < 0:
            raise PlaxisValidationError(f"荷载值必须为非负数，当前值: {self.load_value}")
        
        if not isinstance(self.total_cycles, int) or self.total_cycles <= 0:
            raise PlaxisValidationError(f"循环次数必须为正整数，当前值: {self.total_cycles}")
        
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise PlaxisValidationError(f"批处理大小必须为正整数，当前值: {self.batch_size}")
        
        if self.batch_size > self.total_cycles:
            self.logger.warning(f"批处理大小({self.batch_size})大于总循环次数({self.total_cycles})，已调整为{self.total_cycles}")
            self.batch_size = self.total_cycles
        
        self.logger.info(f"参数验证完成 - 荷载值: {self.load_value}, 循环次数: {self.total_cycles}, 批处理大小: {self.batch_size}")
    
    @contextmanager
    def _plaxis_connection(self):
        """PLAXIS连接上下文管理器"""
        try:
            self.logger.info("尝试连接到PLAXIS...")
            # 模拟PLAXIS连接代码
            # from plxscripting.easy import *
            # s_i, g_i = new_server('localhost', 10000, password='')
            # s_o, g_o = new_server('localhost', 10001, password='')
            
            # 为演示目的，使用模拟对象
            self.g_i = MockPlaxisInput()
            self.g_o = MockPlaxisOutput()
            
            self.logger.info("PLAXIS连接成功")
            yield self.g_i, self.g_o
            
        except Exception as e:
            error_msg = f"PLAXIS连接失败: {str(e)}"
            self.logger.error(error_msg)
            raise PlaxisConnectionError(error_msg)
        
        finally:
            # 清理连接
            if hasattr(self, 'g_i') and self.g_i:
                try:
                    # 模拟关闭连接
                    self.g_i = None
                    self.g_o = None
                    self.logger.info("PLAXIS连接已关闭")
                except Exception as e:
                    self.logger.warning(f"关闭PLAXIS连接时出现警告: {str(e)}")
    
    def _activate_elements(self, g_i, stage_name: str) -> bool:
        """
        激活元素的通用函数
        
        Args:
            g_i: PLAXIS输入对象
            stage_name: 阶段名称
            
        Returns:
            bool: 激活成功返回True，失败返回False
        """
        try:
            self.logger.debug(f"激活阶段 {stage_name} 的元素...")
            
            # 模拟激活元素的代码
            # g_i.gotostructures()
            # g_i.activate([g_i.Plate_1, g_i.Plate_2], g_i.Phases[stage_name])
            
            # 为演示目的，使用模拟方法
            g_i.activate_elements(stage_name)
            
            self.logger.debug(f"阶段 {stage_name} 元素激活成功")
            return True
            
        except Exception as e:
            error_msg = f"激活阶段 {stage_name} 元素失败: {str(e)}"
            self.logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return False
    
    def _apply_load(self, g_i, stage_name: str, load_value: float) -> bool:
        """
        应用荷载
        
        Args:
            g_i: PLAXIS输入对象
            stage_name: 阶段名称
            load_value: 荷载值
            
        Returns:
            bool: 应用成功返回True，失败返回False
        """
        try:
            self.logger.debug(f"在阶段 {stage_name} 应用荷载 Qx={load_value}")
            
            # 检查点荷载是否存在
            if not g_i.has_point_load():
                raise PlaxisLoadError("点荷载不存在，请检查模型设置")
            
            # 模拟应用荷载的代码
            # g_i.gotostages()
            # g_i.PointLoad_1_1.Qx[g_i.Phases[stage_name]] = load_value
            
            # 为演示目的，使用模拟方法
            g_i.apply_point_load(stage_name, load_value)
            
            self.logger.debug(f"阶段 {stage_name} 荷载应用成功: Qx={load_value}")
            return True
            
        except PlaxisLoadError:
            raise
        except Exception as e:
            error_msg = f"在阶段 {stage_name} 应用荷载失败: {str(e)}"
            self.logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return False
    
    def _create_stage(self, g_i, stage_number: int) -> bool:
        """
        创建计算阶段
        
        Args:
            g_i: PLAXIS输入对象
            stage_number: 阶段编号
            
        Returns:
            bool: 创建成功返回True，失败返回False
        """
        try:
            stage_name = f"Stage_{stage_number}"
            
            # 确定荷载值：奇数阶段加载，偶数阶段卸载
            load_value = self.load_value if stage_number % 2 == 1 else 0.0
            load_type = "加载" if stage_number % 2 == 1 else "卸载"
            
            self.logger.debug(f"创建阶段 {stage_number}: {load_type} (Qx={load_value})")
            
            # 模拟创建阶段的代码
            # g_i.gotostages()
            # g_i.phase(g_i.InitialPhase)
            
            # 为演示目的，使用模拟方法
            g_i.create_stage(stage_name)
            
            # 激活元素
            if not self._activate_elements(g_i, stage_name):
                return False
            
            # 应用荷载
            if not self._apply_load(g_i, stage_name, load_value):
                return False
            
            self.logger.info(f"阶段 {stage_number} 创建成功: {load_type} (Qx={load_value})")
            self.stats['completed_stages'] += 1
            return True
            
        except Exception as e:
            error_msg = f"创建阶段 {stage_number} 失败: {str(e)}"
            self.logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            self.stats['failed_stages'] += 1
            return False
    
    def _process_batch(self, g_i, start_stage: int, end_stage: int) -> int:
        """
        批量处理阶段
        
        Args:
            g_i: PLAXIS输入对象
            start_stage: 起始阶段
            end_stage: 结束阶段
            
        Returns:
            int: 成功创建的阶段数
        """
        successful_stages = 0
        batch_start_time = time.time()
        
        self.logger.info(f"开始批量处理阶段 {start_stage} 到 {end_stage}")
        
        for stage_num in range(start_stage, end_stage + 1):
            try:
                if self._create_stage(g_i, stage_num):
                    successful_stages += 1
                
                # 每10个阶段报告一次进度
                if stage_num % 10 == 0:
                    progress = (stage_num / self.total_cycles) * 100
                    self.logger.info(f"进度: {progress:.1f}% ({stage_num}/{self.total_cycles})")
                
            except KeyboardInterrupt:
                self.logger.warning(f"用户中断，已完成到阶段 {stage_num-1}")
                break
            except Exception as e:
                self.logger.error(f"批处理阶段 {stage_num} 时发生未预期错误: {str(e)}")
                continue
        
        batch_time = time.time() - batch_start_time
        self.logger.info(f"批量处理完成: {successful_stages} 个阶段成功，耗时 {batch_time:.2f} 秒")
        
        return successful_stages
    
    def run_cyclic_loading(self) -> Dict[str, Any]:
        """
        执行循环加载
        
        Returns:
            Dict[str, Any]: 执行结果统计
        """
        self.stats['start_time'] = time.time()
        self.logger.info(f"开始PLAXIS 3D循环加载: {self.total_cycles}个循环，批处理大小: {self.batch_size}")
        
        try:
            with self._plaxis_connection() as (g_i, g_o):
                # 批量处理阶段
                current_stage = 1
                while current_stage <= self.total_cycles:
                    end_stage = min(current_stage + self.batch_size - 1, self.total_cycles)
                    
                    successful_count = self._process_batch(g_i, current_stage, end_stage)
                    
                    if successful_count == 0:
                        self.logger.warning(f"批次 {current_stage}-{end_stage} 中没有成功的阶段")
                    
                    current_stage = end_stage + 1
                    
                    # 内存优化：每个批次后强制垃圾回收
                    import gc
                    gc.collect()
        
        except PlaxisConnectionError:
            self.logger.error("PLAXIS连接失败，无法执行循环加载")
            raise
        except KeyboardInterrupt:
            self.logger.info("用户中断执行")
        except Exception as e:
            self.logger.error(f"执行循环加载时发生未预期错误: {str(e)}")
            raise
        
        finally:
            self.stats['end_time'] = time.time()
            self._generate_report()
        
        return self.stats
    
    def _generate_report(self) -> None:
        """生成执行报告"""
        if self.stats['start_time'] and self.stats['end_time']:
            total_time = self.stats['end_time'] - self.stats['start_time']
            success_rate = (self.stats['completed_stages'] / self.total_cycles) * 100
            
            self.logger.info("=" * 60)
            self.logger.info("PLAXIS 3D 循环加载执行报告")
            self.logger.info("=" * 60)
            self.logger.info(f"总循环次数: {self.total_cycles}")
            self.logger.info(f"成功完成: {self.stats['completed_stages']}")
            self.logger.info(f"失败次数: {self.stats['failed_stages']}")
            self.logger.info(f"成功率: {success_rate:.2f}%")
            self.logger.info(f"总耗时: {total_time:.2f} 秒")
            self.logger.info(f"平均每阶段耗时: {total_time/max(self.stats['completed_stages'], 1):.3f} 秒")
            
            if self.stats['errors']:
                self.logger.info(f"错误汇总 (共{len(self.stats['errors'])}个):")
                for i, error in enumerate(self.stats['errors'][:5], 1):  # 只显示前5个错误
                    self.logger.info(f"  {i}. {error}")
                if len(self.stats['errors']) > 5:
                    self.logger.info(f"  ... 还有 {len(self.stats['errors']) - 5} 个错误")
            
            self.logger.info("=" * 60)


# 模拟PLAXIS对象用于演示
class MockPlaxisInput:
    """模拟PLAXIS输入对象"""
    
    def __init__(self):
        self.stages = {}
        self.point_loads_exist = True
    
    def has_point_load(self) -> bool:
        return self.point_loads_exist
    
    def create_stage(self, stage_name: str) -> None:
        self.stages[stage_name] = {'activated': False, 'load_applied': False}
    
    def activate_elements(self, stage_name: str) -> None:
        if stage_name in self.stages:
            self.stages[stage_name]['activated'] = True
    
    def apply_point_load(self, stage_name: str, load_value: float) -> None:
        if stage_name in self.stages:
            self.stages[stage_name]['load_applied'] = True
            self.stages[stage_name]['load_value'] = load_value


class MockPlaxisOutput:
    """模拟PLAXIS输出对象"""
    pass


def main():
    """主函数"""
    try:
        # 创建循环加载器
        loader = PlaxisCyclicLoader(
            load_value=100.0,
            total_cycles=1000,
            batch_size=50,
            log_level='INFO'
        )
        
        # 执行循环加载
        results = loader.run_cyclic_loading()
        
        print("\n执行完成！详细信息请查看日志文件 'plaxis_cyclic_loading.log'")
        return 0
        
    except PlaxisValidationError as e:
        print(f"参数验证错误: {e}")
        return 1
    except PlaxisConnectionError as e:
        print(f"PLAXIS连接错误: {e}")
        return 2
    except Exception as e:
        print(f"未预期错误: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())