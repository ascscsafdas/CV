"""
Scientific Journal Plotting Library
====================================

This module provides a comprehensive solution for creating publication-quality
scientific plots that meet SCI journal standards.

Features:
- 600 DPI resolution for publication quality
- Color-blind friendly palettes
- Cross-platform font compatibility
- Multiple output formats (PNG, SVG, PDF, EPS)
- Configurable style templates
- Statistical significance annotations
- Data validation and preprocessing
- Batch processing capabilities

Author: Scientific Plotting Team
License: MIT
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
import warnings
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats for scientific plots."""
    PNG = 'png'
    SVG = 'svg'
    PDF = 'pdf'
    EPS = 'eps'
    

class ColorPalette(Enum):
    """Available color palettes optimized for scientific publications."""
    COLORBLIND = 'colorblind'
    VIRIDIS = 'viridis'
    NATURE = 'tab10'
    SCIENTIFIC = 'Set2'
    GRAYSCALE = 'gray'


@dataclass
class PlotConfig:
    """Configuration class for scientific plotting parameters."""
    
    # Figure dimensions and DPI
    figure_width: float = 8.0  # inches
    figure_height: float = 6.0  # inches
    dpi: int = 600  # SCI journal standard
    
    # Font settings
    font_family: str = 'serif'
    font_serif: List[str] = field(default_factory=lambda: ['Times New Roman', 'DejaVu Serif'])
    title_size: int = 16
    label_size: int = 14
    tick_size: int = 12
    legend_size: int = 12
    
    # Line and marker settings
    line_width: float = 2.0
    marker_size: float = 6.0
    
    # Color and style
    color_palette: ColorPalette = ColorPalette.COLORBLIND
    style: str = 'seaborn-v0_8-whitegrid'
    
    # Grid and spines
    grid_alpha: float = 0.3
    grid_linewidth: float = 0.8
    spine_linewidth: float = 1.2
    
    # Layout
    tight_layout: bool = True
    subplot_adjust_hspace: float = 0.3
    subplot_adjust_wspace: float = 0.3
    
    # Statistical annotations
    significance_level: float = 0.05
    annotation_fontsize: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            elif isinstance(field_value, list):
                config_dict[field_name] = field_value.copy()
            else:
                config_dict[field_name] = field_value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PlotConfig':
        """Create configuration from dictionary."""
        # Handle enum fields
        if 'color_palette' in config_dict:
            config_dict['color_palette'] = ColorPalette(config_dict['color_palette'])
        return cls(**config_dict)
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: Union[str, Path]) -> 'PlotConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)


class DataValidator:
    """Utility class for data validation and preprocessing."""
    
    @staticmethod
    def validate_array_data(data: Union[List, np.ndarray, pd.Series], 
                          name: str = "data") -> np.ndarray:
        """Validate and convert input data to numpy array."""
        if data is None:
            raise ValueError(f"{name} cannot be None")
        
        # Convert to numpy array
        if isinstance(data, (list, pd.Series)):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"{name} must be array-like (list, numpy array, or pandas Series)")
        
        # Check for empty data
        if len(data) == 0:
            raise ValueError(f"{name} cannot be empty")
        
        # Check for all NaN values
        if np.all(np.isnan(data)):
            raise ValueError(f"{name} contains only NaN values")
        
        # Log warnings for NaN values
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            logger.warning(f"{name} contains {nan_count} NaN values")
        
        return data
    
    @staticmethod
    def validate_equal_length(*arrays) -> None:
        """Validate that all arrays have the same length."""
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            raise ValueError(f"All arrays must have the same length. Got lengths: {lengths}")
    
    @staticmethod
    def preprocess_data(data: np.ndarray, 
                       remove_outliers: bool = False,
                       outlier_method: str = 'iqr',
                       outlier_factor: float = 1.5) -> np.ndarray:
        """Preprocess data with optional outlier removal."""
        if not remove_outliers:
            return data
        
        if outlier_method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_factor * IQR
            upper_bound = Q3 + outlier_factor * IQR
            return data[(data >= lower_bound) & (data <= upper_bound)]
        elif outlier_method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return data[z_scores < outlier_factor]
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")


class ScientificPlotter:
    """
    Main class for creating scientific plots meeting SCI journal standards.
    
    This class provides methods for creating various types of scientific plots
    with publication-quality formatting, color-blind friendly palettes, and
    proper statistical annotations.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize the ScientificPlotter.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or PlotConfig()
        self.validator = DataValidator()
        self._setup_matplotlib()
        self._color_palette = self._get_color_palette()
    
    def _setup_matplotlib(self) -> None:
        """Configure matplotlib with journal-standard settings."""
        try:
            plt.style.use(self.config.style)
        except OSError:
            logger.warning(f"Style '{self.config.style}' not found, using default")
            plt.style.use('default')
        
        # Set font parameters
        plt.rcParams['font.family'] = self.config.font_family
        plt.rcParams['font.serif'] = self.config.font_serif
        plt.rcParams['axes.titlesize'] = self.config.title_size
        plt.rcParams['axes.labelsize'] = self.config.label_size
        plt.rcParams['xtick.labelsize'] = self.config.tick_size
        plt.rcParams['ytick.labelsize'] = self.config.tick_size
        plt.rcParams['legend.fontsize'] = self.config.legend_size
        plt.rcParams['lines.linewidth'] = self.config.line_width
        plt.rcParams['lines.markersize'] = self.config.marker_size
        
        # Set figure parameters
        plt.rcParams['figure.figsize'] = (self.config.figure_width, self.config.figure_height)
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['savefig.dpi'] = self.config.dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
        
        # Grid settings
        plt.rcParams['grid.alpha'] = self.config.grid_alpha
        plt.rcParams['grid.linewidth'] = self.config.grid_linewidth
        plt.rcParams['axes.linewidth'] = self.config.spine_linewidth
        
        # Improve text rendering
        plt.rcParams['text.usetex'] = False  # Avoid LaTeX dependencies
        plt.rcParams['mathtext.default'] = 'regular'
        
        logger.info("Matplotlib configured for scientific plotting")
    
    def _get_color_palette(self) -> List[str]:
        """Get the configured color palette."""
        if self.config.color_palette == ColorPalette.COLORBLIND:
            return sns.color_palette("colorblind").as_hex()
        elif self.config.color_palette == ColorPalette.VIRIDIS:
            return sns.color_palette("viridis", 10).as_hex()
        elif self.config.color_palette == ColorPalette.NATURE:
            return sns.color_palette("tab10").as_hex()
        elif self.config.color_palette == ColorPalette.SCIENTIFIC:
            return sns.color_palette("Set2").as_hex()
        elif self.config.color_palette == ColorPalette.GRAYSCALE:
            return sns.color_palette("gray", 10).as_hex()
        else:
            return sns.color_palette("colorblind").as_hex()
    
    def plot_time_series_forecast(self,
                                actual: Union[List, np.ndarray, pd.Series],
                                predicted: Union[List, np.ndarray, pd.Series],
                                time_index: Optional[Union[List, np.ndarray, pd.Series]] = None,
                                title: str = 'Time Series Forecast',
                                xlabel: str = 'Time',
                                ylabel: str = 'Value',
                                show_metrics: bool = True,
                                save_path: Optional[Union[str, Path]] = None,
                                output_formats: Optional[List[OutputFormat]] = None) -> plt.Figure:
        """
        Create a time series forecast plot with publication-quality formatting.
        
        Args:
            actual: Actual values
            predicted: Predicted values  
            time_index: Time index for x-axis. If None, uses range indices.
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_metrics: Whether to display performance metrics
            save_path: Path to save the plot (without extension)
            output_formats: List of output formats to save
            
        Returns:
            matplotlib Figure object
        """
        # Validate input data
        actual = self.validator.validate_array_data(actual, "actual")
        predicted = self.validator.validate_array_data(predicted, "predicted")
        self.validator.validate_equal_length(actual, predicted)
        
        if time_index is None:
            time_index = np.arange(len(actual))
        else:
            time_index = self.validator.validate_array_data(time_index, "time_index")
            self.validator.validate_equal_length(actual, predicted, time_index)
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        
        # Plot data with scientific styling
        ax.plot(time_index, actual, 
               color=self._color_palette[0], 
               label='Actual', 
               linewidth=self.config.line_width,
               alpha=0.9)
        ax.plot(time_index, predicted, 
               color=self._color_palette[1], 
               label='Predicted', 
               linewidth=self.config.line_width,
               alpha=0.9)
        
        # Formatting
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.config.label_size)
        ax.set_ylabel(ylabel, fontsize=self.config.label_size)
        
        # Legend with scientific formatting
        legend = ax.legend(fontsize=self.config.legend_size, 
                          frameon=True, 
                          fancybox=True, 
                          shadow=True,
                          loc='best')
        legend.get_frame().set_alpha(0.9)
        
        # Add performance metrics if requested
        if show_metrics:
            self._add_performance_metrics(ax, actual, predicted)
        
        # Grid and spines
        ax.grid(True, alpha=self.config.grid_alpha, linewidth=self.config.grid_linewidth)
        for spine in ax.spines.values():
            spine.set_linewidth(self.config.spine_linewidth)
        
        # Layout
        if self.config.tight_layout:
            plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path, output_formats)
        
        return fig
    
    def plot_forecast_with_intervals(self,
                                   actual: Union[List, np.ndarray, pd.Series],
                                   predicted: Union[List, np.ndarray, pd.Series],
                                   lower_bound: Union[List, np.ndarray, pd.Series],
                                   upper_bound: Union[List, np.ndarray, pd.Series],
                                   time_index: Optional[Union[List, np.ndarray, pd.Series]] = None,
                                   title: str = 'Forecast with Confidence Intervals',
                                   xlabel: str = 'Time',
                                   ylabel: str = 'Value',
                                   confidence_level: float = 0.95,
                                   show_metrics: bool = True,
                                   save_path: Optional[Union[str, Path]] = None,
                                   output_formats: Optional[List[OutputFormat]] = None) -> plt.Figure:
        """
        Create a forecast plot with confidence intervals.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            lower_bound: Lower confidence bound
            upper_bound: Upper confidence bound
            time_index: Time index for x-axis
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            confidence_level: Confidence level for intervals
            show_metrics: Whether to display performance metrics
            save_path: Path to save the plot
            output_formats: List of output formats to save
            
        Returns:
            matplotlib Figure object
        """
        # Validate input data
        actual = self.validator.validate_array_data(actual, "actual")
        predicted = self.validator.validate_array_data(predicted, "predicted")
        lower_bound = self.validator.validate_array_data(lower_bound, "lower_bound")
        upper_bound = self.validator.validate_array_data(upper_bound, "upper_bound")
        self.validator.validate_equal_length(actual, predicted, lower_bound, upper_bound)
        
        if time_index is None:
            time_index = np.arange(len(actual))
        else:
            time_index = self.validator.validate_array_data(time_index, "time_index")
            self.validator.validate_equal_length(actual, predicted, time_index)
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        
        # Plot main data
        ax.plot(time_index, actual,
               color=self._color_palette[0],
               label='Actual',
               linewidth=self.config.line_width,
               alpha=0.9)
        ax.plot(time_index, predicted,
               color=self._color_palette[1],
               label='Predicted',
               linewidth=self.config.line_width,
               alpha=0.9)
        
        # Add confidence interval
        ax.fill_between(time_index, lower_bound, upper_bound,
                       color=self._color_palette[2],
                       alpha=0.3,
                       label=f'{confidence_level*100:.0f}% Confidence Interval')
        
        # Formatting
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.config.label_size)
        ax.set_ylabel(ylabel, fontsize=self.config.label_size)
        
        # Legend
        legend = ax.legend(fontsize=self.config.legend_size,
                          frameon=True,
                          fancybox=True,
                          shadow=True,
                          loc='best')
        legend.get_frame().set_alpha(0.9)
        
        # Add performance metrics if requested
        if show_metrics:
            self._add_performance_metrics(ax, actual, predicted)
        
        # Grid and spines
        ax.grid(True, alpha=self.config.grid_alpha, linewidth=self.config.grid_linewidth)
        for spine in ax.spines.values():
            spine.set_linewidth(self.config.spine_linewidth)
        
        # Layout
        if self.config.tight_layout:
            plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path, output_formats)
        
        return fig
    
    def plot_model_comparison(self,
                            data_dict: Dict[str, Dict[str, Union[List, np.ndarray, pd.Series]]],
                            metrics: Optional[List[str]] = None,
                            title: str = 'Model Performance Comparison',
                            save_path: Optional[Union[str, Path]] = None,
                            output_formats: Optional[List[OutputFormat]] = None) -> plt.Figure:
        """
        Create a comprehensive model comparison plot.
        
        Args:
            data_dict: Dictionary with model names as keys and data dictionaries as values
                      Expected format: {'Model1': {'actual': [...], 'predicted': [...]}, ...}
            metrics: List of metrics to display ['rmse', 'mae', 'r2']
            title: Plot title
            save_path: Path to save the plot
            output_formats: List of output formats to save
            
        Returns:
            matplotlib Figure object
        """
        if not data_dict:
            raise ValueError("data_dict cannot be empty")
        
        n_models = len(data_dict)
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.config.figure_width * 1.5, 
                                               self.config.figure_height * 1.5))
        axes = axes.flatten()
        
        # Colors for different models
        colors = self._color_palette[:n_models]
        
        # Plot 1: Time series comparison
        ax = axes[0]
        for i, (model_name, data) in enumerate(data_dict.items()):
            actual = self.validator.validate_array_data(data['actual'], f"{model_name}_actual")
            predicted = self.validator.validate_array_data(data['predicted'], f"{model_name}_predicted")
            time_index = np.arange(len(actual))
            
            if i == 0:  # Only plot actual once
                ax.plot(time_index, actual, color='black', label='Actual', 
                       linewidth=self.config.line_width, alpha=0.8)
            ax.plot(time_index, predicted, color=colors[i], label=f'{model_name} Predicted',
                   linewidth=self.config.line_width, alpha=0.8)
        
        ax.set_title('Time Series Comparison', fontsize=self.config.title_size)
        ax.set_xlabel('Time', fontsize=self.config.label_size)
        ax.set_ylabel('Value', fontsize=self.config.label_size)
        ax.legend(fontsize=self.config.legend_size)
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Plot 2: Scatter plot (Actual vs Predicted)
        ax = axes[1]
        for i, (model_name, data) in enumerate(data_dict.items()):
            actual = self.validator.validate_array_data(data['actual'], f"{model_name}_actual")
            predicted = self.validator.validate_array_data(data['predicted'], f"{model_name}_predicted")
            
            ax.scatter(actual, predicted, color=colors[i], label=model_name,
                      alpha=0.6, s=self.config.marker_size * 8)
        
        # Perfect prediction line
        all_values = np.concatenate([data['actual'] for data in data_dict.values()])
        min_val, max_val = np.min(all_values), np.max(all_values)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_title('Actual vs Predicted', fontsize=self.config.title_size)
        ax.set_xlabel('Actual', fontsize=self.config.label_size)
        ax.set_ylabel('Predicted', fontsize=self.config.label_size)
        ax.legend(fontsize=self.config.legend_size)
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Plot 3: Residuals
        ax = axes[2]
        for i, (model_name, data) in enumerate(data_dict.items()):
            actual = self.validator.validate_array_data(data['actual'], f"{model_name}_actual")
            predicted = self.validator.validate_array_data(data['predicted'], f"{model_name}_predicted")
            residuals = actual - predicted
            
            ax.scatter(predicted, residuals, color=colors[i], label=model_name,
                      alpha=0.6, s=self.config.marker_size * 8)
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.8)
        ax.set_title('Residual Plot', fontsize=self.config.title_size)
        ax.set_xlabel('Predicted', fontsize=self.config.label_size)
        ax.set_ylabel('Residuals', fontsize=self.config.label_size)
        ax.legend(fontsize=self.config.legend_size)
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Plot 4: Performance metrics
        ax = axes[3]
        model_names = list(data_dict.keys())
        metric_values = {metric: [] for metric in metrics}
        
        for model_name, data in data_dict.items():
            actual = self.validator.validate_array_data(data['actual'], f"{model_name}_actual")
            predicted = self.validator.validate_array_data(data['predicted'], f"{model_name}_predicted")
            
            if 'rmse' in metrics:
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                metric_values['rmse'].append(rmse)
            if 'mae' in metrics:
                mae = np.mean(np.abs(actual - predicted))
                metric_values['mae'].append(mae)
            if 'r2' in metrics:
                ss_res = np.sum((actual - predicted) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                metric_values['r2'].append(r2)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            color_idx = i % len(colors)  # Ensure we don't exceed color palette
            ax.bar(x + i * width, metric_values[metric], width, 
                  label=metric.upper(), color=colors[color_idx], alpha=0.8)
        
        ax.set_title('Performance Metrics', fontsize=self.config.title_size)
        ax.set_xlabel('Models', fontsize=self.config.label_size)
        ax.set_ylabel('Metric Value', fontsize=self.config.label_size)
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names)
        ax.legend(fontsize=self.config.legend_size)
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Overall title
        fig.suptitle(title, fontsize=self.config.title_size + 2, fontweight='bold')
        
        # Adjust layout
        plt.subplots_adjust(hspace=self.config.subplot_adjust_hspace, 
                           wspace=self.config.subplot_adjust_wspace)
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path, output_formats)
        
        return fig
    
    def _add_performance_metrics(self, ax: plt.Axes, 
                               actual: np.ndarray, 
                               predicted: np.ndarray) -> None:
        """Add performance metrics text box to the plot."""
        # Calculate metrics
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Create metrics text
        metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
        
        # Add text box
        ax.text(0.02, 0.98, metrics_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=self.config.annotation_fontsize)
    
    def _save_figure(self, fig: plt.Figure, 
                    base_path: Union[str, Path],
                    output_formats: Optional[List[OutputFormat]] = None) -> None:
        """Save figure in multiple formats."""
        if output_formats is None:
            output_formats = [OutputFormat.PNG]
        
        base_path = Path(base_path)
        
        for fmt in output_formats:
            filepath = base_path.with_suffix(f'.{fmt.value}')
            fig.savefig(filepath, 
                       format=fmt.value,
                       dpi=self.config.dpi,
                       bbox_inches='tight',
                       pad_inches=0.1,
                       facecolor='white',
                       edgecolor='none')
            logger.info(f"Figure saved as {filepath}")
    
    def create_summary_statistics(self, data_dict: Dict[str, Dict[str, Union[List, np.ndarray, pd.Series]]]) -> pd.DataFrame:
        """
        Create a summary statistics table for model comparison.
        
        Args:
            data_dict: Dictionary with model data
            
        Returns:
            DataFrame with summary statistics
        """
        stats_data = []
        
        for model_name, data in data_dict.items():
            actual = self.validator.validate_array_data(data['actual'], f"{model_name}_actual")
            predicted = self.validator.validate_array_data(data['predicted'], f"{model_name}_predicted")
            
            # Calculate statistics
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            stats_data.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE (%)': mape,
                'R²': r2,
                'Data Points': len(actual)
            })
        
        return pd.DataFrame(stats_data)


# Utility functions for backward compatibility and convenience

def plot_time_series_forecast(actual: Union[List, np.ndarray, pd.Series],
                            predicted: Union[List, np.ndarray, pd.Series],
                            title: str = 'Time Series Forecast',
                            xlabel: str = 'Time',
                            ylabel: str = 'Value',
                            config: Optional[PlotConfig] = None,
                            save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Convenience function for creating time series forecast plots.
    """
    plotter = ScientificPlotter(config)
    return plotter.plot_time_series_forecast(
        actual=actual, predicted=predicted, title=title,
        xlabel=xlabel, ylabel=ylabel, save_path=save_path
    )


def plot_forecast_with_intervals(actual: Union[List, np.ndarray, pd.Series],
                               predicted: Union[List, np.ndarray, pd.Series],
                               lower_bound: Union[List, np.ndarray, pd.Series],
                               upper_bound: Union[List, np.ndarray, pd.Series],
                               title: str = 'Forecast with Intervals',
                               xlabel: str = 'Time',
                               ylabel: str = 'Value',
                               config: Optional[PlotConfig] = None,
                               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Convenience function for creating forecast plots with confidence intervals.
    """
    plotter = ScientificPlotter(config)
    return plotter.plot_forecast_with_intervals(
        actual=actual, predicted=predicted, lower_bound=lower_bound,
        upper_bound=upper_bound, title=title, xlabel=xlabel, ylabel=ylabel,
        save_path=save_path
    )


# Example usage and configuration templates
def create_default_config() -> PlotConfig:
    """Create default configuration for SCI journal standards."""
    return PlotConfig()


def create_nature_config() -> PlotConfig:
    """Create configuration optimized for Nature journal standards."""
    config = PlotConfig()
    config.figure_width = 8.5
    config.figure_height = 6.4
    config.color_palette = ColorPalette.NATURE
    config.title_size = 18
    config.label_size = 16
    return config


def create_plos_config() -> PlotConfig:
    """Create configuration optimized for PLOS journal standards."""
    config = PlotConfig()
    config.figure_width = 7.5
    config.figure_height = 5.6
    config.color_palette = ColorPalette.SCIENTIFIC
    config.dpi = 600
    return config


if __name__ == "__main__":
    # Example usage
    logger.info("Scientific Journal Plotting Library loaded successfully")
    
    # Generate sample data for demonstration
    np.random.seed(42)
    time_points = np.arange(100)
    actual_data = np.sin(time_points * 0.1) + np.random.normal(0, 0.1, 100)
    predicted_data = np.sin(time_points * 0.1) + np.random.normal(0, 0.15, 100)
    
    # Create plotter instance
    plotter = ScientificPlotter()
    
    # Create example plot
    fig = plotter.plot_time_series_forecast(
        actual=actual_data,
        predicted=predicted_data,
        title="Example Time Series Forecast",
        xlabel="Time Step",
        ylabel="Signal Value"
    )
    
    print("Example plot created successfully!")
    print("Available color palettes:", [p.value for p in ColorPalette])
    print("Available output formats:", [f.value for f in OutputFormat])