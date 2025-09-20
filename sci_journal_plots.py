import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for journal-style formatting
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)  # Standardized figure size
plt.rcParams['lines.linewidth'] = 2

# Define a professional color palette
color_palette = sns.color_palette("colorblind")

def plot_time_series_forecast(actual, predicted, title='Time Series Forecast', xlabel='Time', ylabel='Value'):
    plt.figure(figsize=plt.rcParams['figure.figsize'])
    plt.plot(actual, color=color_palette[0], label='Actual', alpha=0.8)
    plt.plot(predicted, color=color_palette[1], label='Predicted', alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_forecast_with_intervals(actual, predicted, lower_bound, upper_bound, title='Forecast with Intervals', xlabel='Time', ylabel='Value'):
    plt.figure(figsize=plt.rcParams['figure.figsize'])
    plt.plot(actual, color=color_palette[0], label='Actual', alpha=0.8)
    plt.plot(predicted, color=color_palette[1], label='Predicted', alpha=0.8)
    plt.fill_between(range(len(predicted)), lower_bound, upper_bound, color=color_palette[2], alpha=0.3, label='Confidence Interval')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()