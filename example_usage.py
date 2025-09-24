#!/usr/bin/env python3
"""
Example Usage of Scientific Journal Plotting Library
====================================================

This script demonstrates various usage patterns of the optimized scientific
plotting library, showcasing publication-quality plots that meet SCI journal standards.

Features demonstrated:
- Basic time series forecasting plots
- Forecast plots with confidence intervals
- Model comparison plots
- Different configuration templates
- Multiple output formats
- Data validation and preprocessing

Author: Scientific Plotting Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from optimized_sci_journal_plots import (
    ScientificPlotter, PlotConfig, ColorPalette, OutputFormat,
    create_default_config, create_nature_config, create_plos_config,
    plot_time_series_forecast, plot_forecast_with_intervals
)

def generate_sample_data(n_points=100, noise_level=0.1, seed=42):
    """Generate sample time series data for demonstration."""
    np.random.seed(seed)
    
    # Generate time index
    time_points = np.arange(n_points)
    
    # Generate true signal (combination of sine waves and trend)
    true_signal = (np.sin(time_points * 0.1) + 
                  0.5 * np.sin(time_points * 0.05) + 
                  0.02 * time_points)
    
    # Add noise to create "actual" data
    actual_data = true_signal + np.random.normal(0, noise_level, n_points)
    
    # Create predicted data with different noise characteristics
    predicted_data = true_signal + np.random.normal(0, noise_level * 1.2, n_points)
    
    # Generate confidence intervals
    prediction_std = noise_level * 1.5
    lower_bound = predicted_data - 1.96 * prediction_std
    upper_bound = predicted_data + 1.96 * prediction_std
    
    return {
        'time': time_points,
        'actual': actual_data,
        'predicted': predicted_data,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

def example_basic_plots():
    """Demonstrate basic plotting functionality."""
    print("Creating basic plots...")
    
    # Generate sample data
    data = generate_sample_data()
    
    # Create a scientific plotter with default configuration
    plotter = ScientificPlotter()
    
    # Example 1: Basic time series forecast plot
    fig1 = plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        time_index=data['time'],
        title="Basic Time Series Forecast Example",
        xlabel="Time Steps",
        ylabel="Signal Value",
        save_path="/tmp/basic_forecast"
    )
    
    # Example 2: Forecast with confidence intervals
    fig2 = plotter.plot_forecast_with_intervals(
        actual=data['actual'],
        predicted=data['predicted'],
        lower_bound=data['lower_bound'],
        upper_bound=data['upper_bound'],
        time_index=data['time'],
        title="Forecast with 95% Confidence Intervals",
        xlabel="Time Steps",
        ylabel="Signal Value",
        confidence_level=0.95,
        save_path="/tmp/forecast_intervals"
    )
    
    print("Basic plots created successfully!")
    return fig1, fig2

def example_configuration_templates():
    """Demonstrate different configuration templates."""
    print("Creating plots with different journal configurations...")
    
    data = generate_sample_data()
    
    # Nature journal style
    nature_config = create_nature_config()
    nature_plotter = ScientificPlotter(nature_config)
    
    fig_nature = nature_plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title="Nature Journal Style Plot",
        save_path="/tmp/nature_style"
    )
    
    # PLOS journal style
    plos_config = create_plos_config()
    plos_plotter = ScientificPlotter(plos_config)
    
    fig_plos = plos_plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title="PLOS Journal Style Plot",
        save_path="/tmp/plos_style"
    )
    
    # Custom configuration
    custom_config = PlotConfig()
    custom_config.color_palette = ColorPalette.VIRIDIS
    custom_config.figure_width = 10
    custom_config.figure_height = 7
    custom_config.dpi = 300  # Lower DPI for faster processing
    
    custom_plotter = ScientificPlotter(custom_config)
    
    fig_custom = custom_plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title="Custom Configuration Plot",
        save_path="/tmp/custom_style"
    )
    
    print("Configuration template plots created successfully!")
    return fig_nature, fig_plos, fig_custom

def example_model_comparison():
    """Demonstrate model comparison functionality."""
    print("Creating model comparison plots...")
    
    # Generate data for multiple models
    base_data = generate_sample_data(n_points=80, seed=42)
    
    # Simulate different model performances
    models_data = {
        'Linear Model': {
            'actual': base_data['actual'],
            'predicted': base_data['predicted'] + np.random.normal(0, 0.05, len(base_data['actual']))
        },
        'Neural Network': {
            'actual': base_data['actual'],
            'predicted': base_data['predicted'] + np.random.normal(0, 0.03, len(base_data['actual']))
        },
        'Random Forest': {
            'actual': base_data['actual'],
            'predicted': base_data['predicted'] + np.random.normal(0, 0.08, len(base_data['actual']))
        }
    }
    
    plotter = ScientificPlotter()
    
    # Create comprehensive model comparison
    fig_comparison = plotter.plot_model_comparison(
        data_dict=models_data,
        title="Model Performance Comparison Study",
        save_path="/tmp/model_comparison"
    )
    
    # Generate summary statistics
    stats_df = plotter.create_summary_statistics(models_data)
    print("\nModel Performance Summary:")
    print(stats_df.to_string(index=False))
    
    # Save statistics to CSV
    stats_df.to_csv("/tmp/model_statistics.csv", index=False)
    
    print("Model comparison plots created successfully!")
    return fig_comparison, stats_df

def example_multiple_formats():
    """Demonstrate saving plots in multiple formats."""
    print("Creating plots with multiple output formats...")
    
    data = generate_sample_data()
    plotter = ScientificPlotter()
    
    # Save in all supported formats
    output_formats = [OutputFormat.PNG, OutputFormat.SVG, OutputFormat.PDF, OutputFormat.EPS]
    
    fig = plotter.plot_forecast_with_intervals(
        actual=data['actual'],
        predicted=data['predicted'],
        lower_bound=data['lower_bound'],
        upper_bound=data['upper_bound'],
        title="Multi-Format Export Example",
        save_path="/tmp/multi_format_plot",
        output_formats=output_formats
    )
    
    print("Multi-format plots created successfully!")
    print("Available formats:", [fmt.value for fmt in output_formats])
    return fig

def example_configuration_management():
    """Demonstrate configuration saving and loading."""
    print("Demonstrating configuration management...")
    
    # Create custom configuration
    custom_config = PlotConfig()
    custom_config.figure_width = 12
    custom_config.figure_height = 8
    custom_config.dpi = 300
    custom_config.color_palette = ColorPalette.SCIENTIFIC
    custom_config.title_size = 20
    custom_config.label_size = 16
    
    # Save configuration
    config_path = "/tmp/custom_plot_config.json"
    custom_config.save_config(config_path)
    
    # Load configuration
    loaded_config = PlotConfig.load_config(config_path)
    
    # Verify configuration
    print(f"Original DPI: {custom_config.dpi}")
    print(f"Loaded DPI: {loaded_config.dpi}")
    print(f"Configuration file saved to: {config_path}")
    
    # Use loaded configuration
    data = generate_sample_data()
    plotter = ScientificPlotter(loaded_config)
    
    fig = plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title="Plot with Loaded Configuration",
        save_path="/tmp/loaded_config_plot"
    )
    
    print("Configuration management example completed successfully!")
    return fig, custom_config

def example_convenience_functions():
    """Demonstrate convenience functions for backward compatibility."""
    print("Demonstrating convenience functions...")
    
    data = generate_sample_data()
    
    # Use convenience functions (similar to original API)
    fig1 = plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title="Convenience Function Example 1",
        save_path="/tmp/convenience_1"
    )
    
    fig2 = plot_forecast_with_intervals(
        actual=data['actual'],
        predicted=data['predicted'],
        lower_bound=data['lower_bound'],
        upper_bound=data['upper_bound'],
        title="Convenience Function Example 2",
        save_path="/tmp/convenience_2"
    )
    
    print("Convenience function examples completed successfully!")
    return fig1, fig2

def example_data_validation():
    """Demonstrate data validation and error handling."""
    print("Demonstrating data validation...")
    
    plotter = ScientificPlotter()
    
    # Example with good data
    good_data = generate_sample_data(n_points=50)
    
    try:
        fig_good = plotter.plot_time_series_forecast(
            actual=good_data['actual'],
            predicted=good_data['predicted'],
            title="Valid Data Example"
        )
        print("✓ Valid data processed successfully")
    except Exception as e:
        print(f"✗ Error with valid data: {e}")
    
    # Example with invalid data (different lengths)
    try:
        fig_bad = plotter.plot_time_series_forecast(
            actual=[1, 2, 3, 4, 5],
            predicted=[1, 2, 3],  # Different length
            title="Invalid Data Example"
        )
    except ValueError as e:
        print(f"✓ Correctly caught validation error: {e}")
    
    # Example with NaN values
    data_with_nan = good_data['actual'].copy()
    data_with_nan[10:15] = np.nan  # Insert some NaN values
    
    try:
        fig_nan = plotter.plot_time_series_forecast(
            actual=data_with_nan,
            predicted=good_data['predicted'],
            title="Data with NaN Values"
        )
        print("✓ Data with NaN values handled gracefully")
    except Exception as e:
        print(f"Error with NaN data: {e}")
    
    print("Data validation examples completed successfully!")

def main():
    """Run all examples."""
    print("=" * 60)
    print("Scientific Journal Plotting Library - Example Usage")
    print("=" * 60)
    
    # Create tmp directory for outputs
    import os
    os.makedirs("/tmp", exist_ok=True)
    
    try:
        # Run all examples
        print("\n1. Basic Plots")
        print("-" * 30)
        example_basic_plots()
        
        print("\n2. Configuration Templates")
        print("-" * 30)
        example_configuration_templates()
        
        print("\n3. Model Comparison")
        print("-" * 30)
        example_model_comparison()
        
        print("\n4. Multiple Output Formats")
        print("-" * 30)
        example_multiple_formats()
        
        print("\n5. Configuration Management")
        print("-" * 30)
        example_configuration_management()
        
        print("\n6. Convenience Functions")
        print("-" * 30)
        example_convenience_functions()
        
        print("\n7. Data Validation")
        print("-" * 30)
        example_data_validation()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the /tmp directory for generated plots and files.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()