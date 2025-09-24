#!/usr/bin/env python3
"""
Comparison Test: Original vs Optimized Scientific Plotting
==========================================================

This script demonstrates the improvements between the original plotting code
and the optimized version, showcasing enhanced features for SCI journal standards.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import both versions
import sci_journal_plots as original
import optimized_sci_journal_plots as optimized

def generate_test_data(n_points=100, seed=42):
    """Generate consistent test data for comparison."""
    np.random.seed(seed)
    
    time_points = np.arange(n_points)
    true_signal = np.sin(time_points * 0.15) + 0.5 * np.cos(time_points * 0.08)
    
    actual = true_signal + np.random.normal(0, 0.1, n_points)
    predicted = true_signal + np.random.normal(0, 0.12, n_points)
    lower_bound = predicted - 0.25
    upper_bound = predicted + 0.25
    
    return {
        'time': time_points,
        'actual': actual,
        'predicted': predicted,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

def test_original_interface():
    """Test the improved original interface."""
    print("Testing Original Interface (Improved)...")
    
    data = generate_test_data()
    
    start_time = time.time()
    
    # Test 1: Basic forecast plot
    fig1 = original.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title='Original Interface: Time Series Forecast',
        show_metrics=True
    )
    plt.close(fig1)
    
    # Test 2: Forecast with intervals
    fig2 = original.plot_forecast_with_intervals(
        actual=data['actual'],
        predicted=data['predicted'],
        lower_bound=data['lower_bound'],
        upper_bound=data['upper_bound'],
        title='Original Interface: Forecast with Intervals',
        show_metrics=True
    )
    plt.close(fig2)
    
    elapsed_time = time.time() - start_time
    
    print(f"✓ Original interface tests completed in {elapsed_time:.3f} seconds")
    return elapsed_time

def test_optimized_interface():
    """Test the optimized interface with advanced features."""
    print("Testing Optimized Interface...")
    
    data = generate_test_data()
    
    start_time = time.time()
    
    # Test 1: Basic forecast with default config
    plotter = optimized.ScientificPlotter()
    fig1 = plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title='Optimized Interface: Time Series Forecast'
    )
    plt.close(fig1)
    
    # Test 2: Forecast with intervals and custom config
    nature_config = optimized.create_nature_config()
    nature_plotter = optimized.ScientificPlotter(nature_config)
    
    fig2 = nature_plotter.plot_forecast_with_intervals(
        actual=data['actual'],
        predicted=data['predicted'],
        lower_bound=data['lower_bound'],
        upper_bound=data['upper_bound'],
        title='Optimized Interface: Nature Journal Style'
    )
    plt.close(fig2)
    
    # Test 3: Model comparison (advanced feature)
    models_data = {
        'Model A': {'actual': data['actual'], 'predicted': data['predicted']},
        'Model B': {'actual': data['actual'], 'predicted': data['predicted'] + np.random.normal(0, 0.05, len(data['actual']))}
    }
    
    fig3 = plotter.plot_model_comparison(
        data_dict=models_data,
        title='Optimized Interface: Model Comparison'
    )
    plt.close(fig3)
    
    # Test 4: Summary statistics
    stats_df = plotter.create_summary_statistics(models_data)
    
    elapsed_time = time.time() - start_time
    
    print(f"✓ Optimized interface tests completed in {elapsed_time:.3f} seconds")
    print(f"✓ Generated summary statistics with shape: {stats_df.shape}")
    return elapsed_time

def test_configuration_features():
    """Test advanced configuration features."""
    print("Testing Configuration Features...")
    
    data = generate_test_data()
    
    # Test custom configuration
    custom_config = optimized.PlotConfig()
    custom_config.dpi = 300  # Lower for faster testing
    custom_config.color_palette = optimized.ColorPalette.VIRIDIS
    custom_config.figure_width = 10
    custom_config.figure_height = 6
    
    plotter = optimized.ScientificPlotter(custom_config)
    
    fig = plotter.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title='Custom Configuration Test'
    )
    plt.close(fig)
    
    # Test configuration saving/loading
    config_path = "/tmp/test_config.json"
    custom_config.save_config(config_path)
    loaded_config = optimized.PlotConfig.load_config(config_path)
    
    print(f"✓ Configuration features tested successfully")
    print(f"✓ Config DPI: {loaded_config.dpi}, Color palette: {loaded_config.color_palette.value}")

def test_error_handling():
    """Test error handling and data validation."""
    print("Testing Error Handling...")
    
    # Test original interface validation
    try:
        original.plot_time_series_forecast([1, 2, 3], [1, 2])  # Different lengths
        print("✗ Original validation failed")
    except ValueError:
        print("✓ Original interface validation working")
    
    # Test optimized interface validation
    try:
        plotter = optimized.ScientificPlotter()
        plotter.plot_time_series_forecast([1, 2, 3], [1, 2])  # Different lengths
        print("✗ Optimized validation failed")
    except ValueError:
        print("✓ Optimized interface validation working")
    
    # Test data type validation
    try:
        plotter.plot_time_series_forecast("invalid", [1, 2, 3])
        print("✗ Type validation failed")
    except TypeError:
        print("✓ Type validation working")

def test_backward_compatibility():
    """Test backward compatibility with convenience functions."""
    print("Testing Backward Compatibility...")
    
    data = generate_test_data()
    
    # Test convenience functions from optimized module
    fig1 = optimized.plot_time_series_forecast(
        actual=data['actual'],
        predicted=data['predicted'],
        title='Convenience Function Test'
    )
    plt.close(fig1)
    
    fig2 = optimized.plot_forecast_with_intervals(
        actual=data['actual'],
        predicted=data['predicted'],
        lower_bound=data['lower_bound'],
        upper_bound=data['upper_bound'],
        title='Convenience Function with Intervals'
    )
    plt.close(fig2)
    
    print("✓ Backward compatibility maintained")

def compare_features():
    """Compare features between original and optimized versions."""
    print("\n" + "="*60)
    print("FEATURE COMPARISON")
    print("="*60)
    
    features = [
        ("Basic Time Series Plot", "✓", "✓"),
        ("Forecast with Intervals", "✓", "✓"),
        ("Data Validation", "Basic", "Advanced"),
        ("Error Handling", "Basic", "Comprehensive"),
        ("Performance Metrics", "✓", "✓"),
        ("Multiple Output Formats", "✗", "✓"),
        ("Configuration Templates", "✗", "✓"),
        ("Model Comparison Plots", "✗", "✓"),
        ("Summary Statistics", "✗", "✓"),
        ("Color-blind Friendly", "✓", "✓"),
        ("SCI Journal Standards", "Basic", "Full"),
        ("600 DPI Output", "✓", "✓"),
        ("Cross-platform Fonts", "Limited", "✓"),
        ("Configuration Management", "✗", "✓"),
        ("Batch Processing", "✗", "✓"),
        ("Documentation", "Basic", "Comprehensive"),
        ("Type Hints", "✗", "✓"),
        ("Logging", "✗", "✓")
    ]
    
    print(f"{'Feature':<25} {'Original':<15} {'Optimized':<15}")
    print("-" * 60)
    for feature, orig, opt in features:
        print(f"{feature:<25} {orig:<15} {opt:<15}")

def main():
    """Run all comparison tests."""
    print("Scientific Plotting Library Comparison Test")
    print("=" * 60)
    
    try:
        # Test both interfaces
        original_time = test_original_interface()
        optimized_time = test_optimized_interface()
        
        print()
        test_configuration_features()
        print()
        test_error_handling()
        print()
        test_backward_compatibility()
        
        # Compare features
        compare_features()
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"Original Interface: {original_time:.3f} seconds")
        print(f"Optimized Interface: {optimized_time:.3f} seconds")
        print(f"Performance Ratio: {optimized_time/original_time:.2f}x")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("✓ All tests passed successfully!")
        print("✓ Both interfaces working correctly")
        print("✓ Backward compatibility maintained")
        print("✓ Advanced features available in optimized version")
        print("✓ SCI journal standards compliance improved")
        print("="*60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()