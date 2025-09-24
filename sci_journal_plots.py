"""
Scientific Journal Plotting Library - Original Interface (Updated)
=================================================================

This module provides backward-compatible functions with enhanced features
for creating publication-quality scientific plots.

Updates:
- Fixed deprecated matplotlib styles
- Improved error handling and data validation
- Enhanced color palettes and typography
- Better cross-platform compatibility
- Support for multiple output formats

For advanced features, use the optimized_sci_journal_plots module.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from typing import Union, List, Optional
from pathlib import Path

# Configure matplotlib for journal-style formatting with updated styles
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Updated from deprecated 'seaborn-whitegrid'
except OSError:
    plt.style.use('default')
    warnings.warn("Using default matplotlib style as seaborn-v0_8-whitegrid not available")

# Font configuration with fallbacks for cross-platform compatibility
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']  # Added fallbacks
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)  # Standardized figure size
plt.rcParams['lines.linewidth'] = 2

# Enhanced figure quality settings
plt.rcParams['figure.dpi'] = 600  # SCI journal standard
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Improved text rendering
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

# Define a professional color palette (color-blind friendly)
color_palette = sns.color_palette("colorblind")

def validate_input_data(actual: Union[List, np.ndarray], 
                       predicted: Union[List, np.ndarray],
                       name_actual: str = "actual",
                       name_predicted: str = "predicted") -> tuple:
    """
    Validate and convert input data to numpy arrays.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        name_actual: Name for actual data (for error messages)
        name_predicted: Name for predicted data (for error messages)
        
    Returns:
        Tuple of validated numpy arrays
        
    Raises:
        ValueError: If data is invalid or arrays have different lengths
        TypeError: If data is not array-like
    """
    # Convert to numpy arrays
    if isinstance(actual, list):
        actual = np.array(actual)
    elif not isinstance(actual, np.ndarray):
        raise TypeError(f"{name_actual} must be array-like (list or numpy array)")
    
    if isinstance(predicted, list):
        predicted = np.array(predicted)
    elif not isinstance(predicted, np.ndarray):
        raise TypeError(f"{name_predicted} must be array-like (list or numpy array)")
    
    # Validate data
    if len(actual) == 0:
        raise ValueError(f"{name_actual} cannot be empty")
    if len(predicted) == 0:
        raise ValueError(f"{name_predicted} cannot be empty")
    
    if len(actual) != len(predicted):
        raise ValueError(f"{name_actual} and {name_predicted} must have the same length. "
                        f"Got {len(actual)} and {len(predicted)}")
    
    # Check for all NaN values
    if np.all(np.isnan(actual)):
        raise ValueError(f"{name_actual} contains only NaN values")
    if np.all(np.isnan(predicted)):
        raise ValueError(f"{name_predicted} contains only NaN values")
    
    # Warn about NaN values
    nan_count_actual = np.sum(np.isnan(actual))
    nan_count_predicted = np.sum(np.isnan(predicted))
    
    if nan_count_actual > 0:
        warnings.warn(f"{name_actual} contains {nan_count_actual} NaN values")
    if nan_count_predicted > 0:
        warnings.warn(f"{name_predicted} contains {nan_count_predicted} NaN values")
    
    return actual, predicted

def calculate_performance_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Calculate performance metrics for model evaluation.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with performance metrics
    """
    # Remove NaN values for calculations
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'mape': np.nan}
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
    mae = np.mean(np.abs(actual_clean - predicted_clean))
    
    # R-squared
    ss_res = np.sum((actual_clean - predicted_clean) ** 2)
    ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    # MAPE (handling division by zero)
    mape = np.mean(np.abs((actual_clean - predicted_clean) / np.where(actual_clean == 0, 1e-10, actual_clean))) * 100
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

def add_performance_metrics_text(ax: plt.Axes, metrics: dict, position: str = 'upper left') -> None:
    """
    Add performance metrics text box to the plot.
    
    Args:
        ax: Matplotlib axes object
        metrics: Dictionary with performance metrics
        position: Position of the text box
    """
    # Create metrics text
    metrics_text = (f"RMSE: {metrics['rmse']:.4f}\n"
                   f"MAE: {metrics['mae']:.4f}\n"
                   f"R²: {metrics['r2']:.4f}")
    
    # Position mapping
    positions = {
        'upper left': (0.02, 0.98),
        'upper right': (0.98, 0.98),
        'lower left': (0.02, 0.02),
        'lower right': (0.98, 0.02)
    }
    
    x, y = positions.get(position, positions['upper left'])
    ha = 'left' if x < 0.5 else 'right'
    va = 'top' if y > 0.5 else 'bottom'
    
    # Add text box with improved styling
    ax.text(x, y, metrics_text,
           transform=ax.transAxes,
           verticalalignment=va,
           horizontalalignment=ha,
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='white', 
                    alpha=0.9,
                    edgecolor='gray',
                    linewidth=0.5),
           fontsize=10,
           family='monospace')

def save_figure(fig: plt.Figure, 
               base_path: Union[str, Path],
               formats: Optional[List[str]] = None,
               **kwargs) -> List[Path]:
    """
    Save figure in multiple formats with high quality settings.
    
    Args:
        fig: Matplotlib figure object
        base_path: Base path for saving (without extension)
        formats: List of formats to save ['png', 'pdf', 'svg', 'eps']
        **kwargs: Additional arguments for savefig
        
    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ['png']
    
    base_path = Path(base_path)
    saved_files = []
    
    # Default savefig parameters
    save_params = {
        'dpi': 600,
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_params.update(kwargs)
    
    for fmt in formats:
        filepath = base_path.with_suffix(f'.{fmt}')
        try:
            fig.savefig(filepath, format=fmt, **save_params)
            saved_files.append(filepath)
            print(f"Figure saved as {filepath}")
        except Exception as e:
            warnings.warn(f"Failed to save figure as {fmt}: {e}")
    
    return saved_files


def plot_time_series_forecast(actual: Union[List, np.ndarray], 
                            predicted: Union[List, np.ndarray],
                            title: str = 'Time Series Forecast', 
                            xlabel: str = 'Time', 
                            ylabel: str = 'Value',
                            time_index: Optional[Union[List, np.ndarray]] = None,
                            show_metrics: bool = True,
                            metrics_position: str = 'upper left',
                            save_path: Optional[Union[str, Path]] = None,
                            save_formats: Optional[List[str]] = None,
                            figsize: Optional[tuple] = None) -> plt.Figure:
    """
    Create a time series forecast plot with enhanced formatting and validation.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        time_index: Time index for x-axis (optional)
        show_metrics: Whether to display performance metrics
        metrics_position: Position of metrics text box
        save_path: Path to save the plot (without extension)
        save_formats: List of formats to save the plot
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Validate input data
    actual, predicted = validate_input_data(actual, predicted)
    
    # Set up time index
    if time_index is None:
        time_index = np.arange(len(actual))
    else:
        time_index = np.array(time_index)
        if len(time_index) != len(actual):
            raise ValueError("time_index must have the same length as actual and predicted data")
    
    # Create figure with specified or default size
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data with enhanced styling
    ax.plot(time_index, actual, 
           color=color_palette[0], 
           label='Actual', 
           linewidth=plt.rcParams['lines.linewidth'],
           alpha=0.9,
           marker='o',
           markersize=3,
           markevery=max(1, len(actual)//20))  # Show markers on subset of points
    
    ax.plot(time_index, predicted, 
           color=color_palette[1], 
           label='Predicted', 
           linewidth=plt.rcParams['lines.linewidth'],
           alpha=0.9,
           marker='s',
           markersize=3,
           markevery=max(1, len(predicted)//20))
    
    # Enhanced formatting
    ax.set_title(title, fontsize=plt.rcParams['axes.titlesize'], fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel(ylabel, fontsize=plt.rcParams['axes.labelsize'])
    
    # Improved legend
    legend = ax.legend(fontsize=plt.rcParams['legend.fontsize'],
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      loc='best')
    legend.get_frame().set_alpha(0.9)
    
    # Add performance metrics
    if show_metrics:
        metrics = calculate_performance_metrics(actual, predicted)
        add_performance_metrics_text(ax, metrics, metrics_position)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)
    
    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_figure(fig, save_path, save_formats)
    
    return fig


def plot_forecast_with_intervals(actual: Union[List, np.ndarray],
                               predicted: Union[List, np.ndarray],
                               lower_bound: Union[List, np.ndarray],
                               upper_bound: Union[List, np.ndarray],
                               title: str = 'Forecast with Intervals',
                               xlabel: str = 'Time',
                               ylabel: str = 'Value',
                               time_index: Optional[Union[List, np.ndarray]] = None,
                               confidence_level: float = 0.95,
                               show_metrics: bool = True,
                               metrics_position: str = 'upper left',
                               save_path: Optional[Union[str, Path]] = None,
                               save_formats: Optional[List[str]] = None,
                               figsize: Optional[tuple] = None) -> plt.Figure:
    """
    Create a forecast plot with confidence intervals and enhanced formatting.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        time_index: Time index for x-axis (optional)
        confidence_level: Confidence level for the intervals
        show_metrics: Whether to display performance metrics
        metrics_position: Position of metrics text box
        save_path: Path to save the plot (without extension)
        save_formats: List of formats to save the plot
        figsize: Figure size tuple (width, height)
        
    Returns:
        matplotlib Figure object
    """
    # Validate input data
    actual, predicted = validate_input_data(actual, predicted)
    
    # Validate bounds
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    
    if len(lower_bound) != len(actual) or len(upper_bound) != len(actual):
        raise ValueError("lower_bound and upper_bound must have the same length as actual and predicted data")
    
    # Set up time index
    if time_index is None:
        time_index = np.arange(len(actual))
    else:
        time_index = np.array(time_index)
        if len(time_index) != len(actual):
            raise ValueError("time_index must have the same length as actual and predicted data")
    
    # Create figure
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confidence interval first (behind other lines)
    ax.fill_between(time_index, lower_bound, upper_bound,
                   color=color_palette[2],
                   alpha=0.3,
                   label=f'{confidence_level*100:.0f}% Confidence Interval')
    
    # Plot main data
    ax.plot(time_index, actual,
           color=color_palette[0],
           label='Actual',
           linewidth=plt.rcParams['lines.linewidth'],
           alpha=0.9,
           marker='o',
           markersize=3,
           markevery=max(1, len(actual)//20))
    
    ax.plot(time_index, predicted,
           color=color_palette[1],
           label='Predicted',
           linewidth=plt.rcParams['lines.linewidth'],
           alpha=0.9,
           marker='s',
           markersize=3,
           markevery=max(1, len(predicted)//20))
    
    # Enhanced formatting
    ax.set_title(title, fontsize=plt.rcParams['axes.titlesize'], fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel(ylabel, fontsize=plt.rcParams['axes.labelsize'])
    
    # Improved legend
    legend = ax.legend(fontsize=plt.rcParams['legend.fontsize'],
                      frameon=True,
                      fancybox=True,
                      shadow=True,
                      loc='best')
    legend.get_frame().set_alpha(0.9)
    
    # Add performance metrics
    if show_metrics:
        metrics = calculate_performance_metrics(actual, predicted)
        add_performance_metrics_text(ax, metrics, metrics_position)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
    ax.set_axisbelow(True)
    
    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        save_figure(fig, save_path, save_formats)
    
    return fig


# Convenience function for quick plotting
def quick_forecast_plot(actual: Union[List, np.ndarray], 
                       predicted: Union[List, np.ndarray],
                       **kwargs) -> plt.Figure:
    """
    Quick convenience function for creating forecast plots.
    
    Args:
        actual: Actual values
        predicted: Predicted values  
        **kwargs: Additional arguments passed to plot_time_series_forecast
        
    Returns:
        matplotlib Figure object
    """
    return plot_time_series_forecast(actual, predicted, **kwargs)


if __name__ == "__main__":
    # Example usage with error handling
    try:
        print("Testing updated scientific plotting functions...")
        
        # Generate sample data
        np.random.seed(42)
        time_points = np.arange(50)
        actual_data = np.sin(time_points * 0.1) + np.random.normal(0, 0.1, 50)
        predicted_data = np.sin(time_points * 0.1) + np.random.normal(0, 0.15, 50)
        
        # Test basic functionality
        fig1 = plot_time_series_forecast(
            actual=actual_data,
            predicted=predicted_data,
            title="Updated Scientific Plot Example",
            xlabel="Time Step",
            ylabel="Signal Value"
        )
        
        print("✓ Basic forecast plot created successfully")
        
        # Test with confidence intervals
        lower_bounds = predicted_data - 0.3
        upper_bounds = predicted_data + 0.3
        
        fig2 = plot_forecast_with_intervals(
            actual=actual_data,
            predicted=predicted_data,
            lower_bound=lower_bounds,
            upper_bound=upper_bounds,
            title="Forecast with Confidence Intervals"
        )
        
        print("✓ Forecast with intervals created successfully")
        print("✓ All updates applied successfully!")
        
        # Test error handling
        try:
            plot_time_series_forecast([1, 2, 3], [1, 2])  # Different lengths
        except ValueError as e:
            print(f"✓ Data validation working: {e}")
        
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()