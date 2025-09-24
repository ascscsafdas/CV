# Scientific Journal Plotting Library

A comprehensive Python library for creating publication-quality scientific plots that meet SCI journal standards.

## Features

### ✨ SCI Journal Standards Compliance
- **600 DPI resolution** for publication quality
- **Cross-platform font compatibility** with Times New Roman fallbacks
- **Standardized figure dimensions** following journal guidelines
- **Professional color schemes** including color-blind friendly palettes
- **Optimized for black and white printing**

### 🎨 Advanced Plotting Capabilities
- Time series forecasting plots with confidence intervals
- Model performance comparison visualizations
- Statistical significance annotations
- Multiple subplot layouts with proper spacing
- Publication-ready legends and axis formatting

### 🔧 Flexible Configuration System
- **Configurable templates** for different journals (Nature, PLOS, etc.)
- **JSON-based configuration** for reproducible settings
- **Multiple color palette options** (colorblind, viridis, nature, scientific, grayscale)
- **Customizable typography** and layout parameters

### 📊 Data Management
- **Robust data validation** with informative error messages
- **Automatic preprocessing** with outlier detection
- **Support for multiple data formats** (lists, numpy arrays, pandas Series)
- **Missing data handling** with warnings and graceful degradation

### 💾 Export Capabilities
- **Multiple output formats**: PNG, SVG, PDF, EPS
- **Batch processing** for multiple plots
- **High-resolution outputs** suitable for journal submission
- **Consistent file naming** and organization

## Installation

The library requires the following dependencies:

```bash
pip install matplotlib seaborn numpy pandas
```

## Quick Start

### Basic Usage

```python
from optimized_sci_journal_plots import ScientificPlotter
import numpy as np

# Generate sample data
actual = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
predicted = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.15, 100)

# Create plotter
plotter = ScientificPlotter()

# Create publication-quality plot
fig = plotter.plot_time_series_forecast(
    actual=actual,
    predicted=predicted,
    title="Time Series Forecast Example",
    xlabel="Time",
    ylabel="Value",
    save_path="forecast_plot"
)
```

### Using Convenience Functions

```python
from optimized_sci_journal_plots import plot_time_series_forecast

# Simple function call (backward compatible)
fig = plot_time_series_forecast(
    actual=actual_data,
    predicted=predicted_data,
    title="My Forecast",
    save_path="my_plot"
)
```

## Configuration System

### Using Predefined Templates

```python
from optimized_sci_journal_plots import (
    ScientificPlotter, 
    create_nature_config, 
    create_plos_config
)

# Nature journal style
nature_config = create_nature_config()
plotter = ScientificPlotter(nature_config)

# PLOS journal style
plos_config = create_plos_config()
plotter = ScientificPlotter(plos_config)
```

### Custom Configuration

```python
from optimized_sci_journal_plots import PlotConfig, ColorPalette

# Create custom configuration
config = PlotConfig()
config.figure_width = 10.0
config.figure_height = 7.0
config.dpi = 600
config.color_palette = ColorPalette.VIRIDIS
config.title_size = 18
config.label_size = 14

# Use custom configuration
plotter = ScientificPlotter(config)
```

### Saving and Loading Configurations

```python
# Save configuration
config.save_config("my_journal_config.json")

# Load configuration
loaded_config = PlotConfig.load_config("my_journal_config.json")
```

## Plot Types

### 1. Time Series Forecasting

```python
fig = plotter.plot_time_series_forecast(
    actual=actual_data,
    predicted=predicted_data,
    time_index=time_points,  # Optional
    title="Time Series Forecast",
    xlabel="Time",
    ylabel="Value",
    show_metrics=True,  # Display RMSE, MAE, R²
    save_path="forecast",
    output_formats=[OutputFormat.PNG, OutputFormat.PDF]
)
```

### 2. Forecasting with Confidence Intervals

```python
fig = plotter.plot_forecast_with_intervals(
    actual=actual_data,
    predicted=predicted_data,
    lower_bound=lower_bounds,
    upper_bound=upper_bounds,
    confidence_level=0.95,
    title="Forecast with 95% Confidence Intervals"
)
```

### 3. Model Comparison

```python
models_data = {
    'Linear Model': {'actual': actual, 'predicted': pred1},
    'Neural Network': {'actual': actual, 'predicted': pred2},
    'Random Forest': {'actual': actual, 'predicted': pred3}
}

fig = plotter.plot_model_comparison(
    data_dict=models_data,
    title="Model Performance Comparison"
)

# Get summary statistics
stats_df = plotter.create_summary_statistics(models_data)
```

## Configuration Parameters

### Figure Settings
- `figure_width`: Figure width in inches (default: 8.0)
- `figure_height`: Figure height in inches (default: 6.0)
- `dpi`: Resolution in dots per inch (default: 600)

### Typography
- `font_family`: Font family (default: 'serif')
- `font_serif`: List of serif fonts (default: ['Times New Roman', 'DejaVu Serif'])
- `title_size`: Title font size (default: 16)
- `label_size`: Axis label font size (default: 14)
- `tick_size`: Tick label font size (default: 12)
- `legend_size`: Legend font size (default: 12)

### Visual Elements
- `line_width`: Line width for plots (default: 2.0)
- `marker_size`: Marker size for scatter plots (default: 6.0)
- `grid_alpha`: Grid transparency (default: 0.3)
- `spine_linewidth`: Axes border width (default: 1.2)

### Color Palettes
- `COLORBLIND`: Color-blind friendly palette
- `VIRIDIS`: Perceptually uniform color scale
- `NATURE`: Nature journal style colors
- `SCIENTIFIC`: Scientific publication colors
- `GRAYSCALE`: Grayscale palette for print compatibility

## Best Practices

### 1. Data Preparation
```python
# Always validate your data
from optimized_sci_journal_plots import DataValidator

validator = DataValidator()

# Validate arrays
actual = validator.validate_array_data(actual_data, "actual")
predicted = validator.validate_array_data(predicted_data, "predicted")

# Check equal lengths
validator.validate_equal_length(actual, predicted)

# Preprocess if needed
clean_data = validator.preprocess_data(
    raw_data, 
    remove_outliers=True, 
    outlier_method='iqr'
)
```

### 2. Publication Standards
- Use **600 DPI** for journal submissions
- Choose **color-blind friendly palettes**
- Include **statistical metrics** (RMSE, MAE, R²)
- Use **consistent typography** throughout figures
- Export to **multiple formats** for different uses

### 3. File Organization
```python
# Organized file naming
base_path = "results/experiment_1/plots"

# Save in multiple formats
output_formats = [
    OutputFormat.PNG,  # For presentations
    OutputFormat.PDF,  # For publications
    OutputFormat.SVG   # For editing
]

fig = plotter.plot_time_series_forecast(
    actual=data['actual'],
    predicted=data['predicted'],
    save_path=f"{base_path}/forecast_analysis",
    output_formats=output_formats
)
```

## Journal-Specific Guidelines

### Nature Family Journals
```python
config = create_nature_config()
# - Figure width: 8.5 inches
# - Figure height: 6.4 inches  
# - Nature color palette
# - Larger font sizes
```

### PLOS Journals
```python
config = create_plos_config()
# - Figure width: 7.5 inches
# - Figure height: 5.6 inches
# - Scientific color palette
# - 600 DPI resolution
```

### Custom Journal Requirements
```python
# Check your target journal's guidelines
config = PlotConfig()
config.figure_width = 7.0  # Journal requirement
config.figure_height = 5.0  # Journal requirement
config.dpi = 300  # Journal requirement
config.color_palette = ColorPalette.GRAYSCALE  # For print
```

## Performance Metrics

The library automatically calculates and displays:

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## Error Handling

The library provides comprehensive error handling:

```python
try:
    fig = plotter.plot_time_series_forecast(actual, predicted)
except ValueError as e:
    print(f"Data validation error: {e}")
except TypeError as e:
    print(f"Data type error: {e}")
```

Common errors and solutions:

1. **Different array lengths**: Ensure all input arrays have the same length
2. **Invalid data types**: Use lists, numpy arrays, or pandas Series
3. **Empty data**: Check that your data arrays are not empty
4. **All NaN values**: Handle missing data before plotting

## Advanced Features

### Statistical Significance Testing
```python
# Future feature - statistical annotations
plotter.add_significance_markers(
    data1, data2, 
    test='t-test', 
    alpha=0.05
)
```

### Batch Processing
```python
# Process multiple datasets
datasets = ['exp1.csv', 'exp2.csv', 'exp3.csv']

for dataset in datasets:
    data = load_data(dataset)
    fig = plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        save_path=f"plots/{dataset}_forecast"
    )
```

## Troubleshooting

### Common Issues

1. **Font not found**: The library falls back to DejaVu Serif if Times New Roman is unavailable
2. **Style not found**: Uses 'default' style if the specified style is unavailable
3. **Low DPI warning**: Check that your DPI setting meets journal requirements
4. **Color palette issues**: Ensure you're using the correct ColorPalette enum values

### Performance Tips

1. Use lower DPI (300) for preview plots
2. Use PNG format for quick iterations
3. Use PDF/SVG for final publications
4. Process data in chunks for large datasets

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic plotting functionality
- Configuration management
- Model comparison plots
- Multiple output formats
- Data validation examples
- Error handling demonstrations

## Contributing

To contribute to this library:

1. Follow PEP 8 coding standards
2. Add type hints to all functions
3. Include comprehensive docstrings
4. Add unit tests for new features
5. Update documentation for changes

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check this documentation
2. Review the example usage
3. Check the error messages (they're designed to be helpful)
4. Refer to matplotlib and seaborn documentation for advanced customization