import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(analysis_forecast_result):
    plot_list = {}

    plot_list['components'] = plot_components(analysis_forecast_result, log=False)
    plot_list['log_components'] = plot_components(analysis_forecast_result, log=True)

    if 'analysis' in analysis_forecast_result.attrs.get('type', []):
        plot_list['original_and_smooth'] = plot_performance(analysis_forecast_result)
    else:
        plot_list['original_and_forecast'] = plot_performance(analysis_forecast_result)

    return plot_list


def plot_components(analysis_forecast_result, log=True):
    if 'analysis' in analysis_forecast_result.attrs.get('type', []):
        title = "Components of Log Intraday Volume (analysis)" if log else "Components of Intraday Volume (analysis)"
    else:
        title = "Components of Log Intraday Volume (forecast)" if log else "Components of Intraday Volume (forecast)"
    
    components = analysis_forecast_result['components']  # Assuming 'components' is a key
    plt_data = pd.DataFrame({
        'original': analysis_forecast_result['original_signal'],
        'daily': components['daily'],
        'seasonal': components['seasonal'],
        'dynamic': components['dynamic'],
        'residual': components['residual']
    })

    if log:
        plt_data = np.log10(plt_data)

    plt_data['i'] = np.arange(1, len(plt_data) + 1)

    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    text_size = 10

    # Plot Original
    axes[0].plot(plt_data['i'], plt_data['original'], color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Original")
    axes[0].set_title("Original Signal")
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Plot Daily
    axes[1].plot(plt_data['i'], plt_data['daily'], color="steelblue", linewidth=1)
    axes[1].set_ylabel("Daily")
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Plot Seasonal
    axes[2].plot(plt_data['i'], plt_data['seasonal'], color="steelblue", linewidth=0.8)
    axes[2].set_ylabel("Seasonal")
    axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Plot Dynamic
    axes[3].plot(plt_data['i'], plt_data['dynamic'], color="steelblue", linewidth=0.8)
    axes[3].set_ylabel("Dynamic")

    # Plot Residual
    axes[4].plot(plt_data['i'], plt_data['residual'], color="steelblue", linewidth=0.8)
    axes[4].set_ylabel("Residual")
    axes[4].set_xlabel("Time (bins)")

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    return fig


def plot_performance(analysis_forecast_result):
    if 'analysis' in analysis_forecast_result.attrs.get('type', []):
        approximated_signal = analysis_forecast_result['smooth_signal']
        title = "Original and Smooth Signals (analysis)"
        legend_name = "smooth"
    else:
        approximated_signal = analysis_forecast_result['forecast_signal']
        title = "Original and One-bin-ahead Forecast signal (forecast)"
        legend_name = "forecast"

    plt_data = pd.DataFrame({
        'original': analysis_forecast_result['original_signal'],
        'output': approximated_signal
    })

    plt_data['i'] = np.arange(1, len(plt_data) + 1)

    plt_reshape = pd.melt(plt_data, id_vars=['i'], var_name="variable", value_name="value")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plt_reshape, x='i', y='value', hue='variable', palette=["steelblue", "#FD6467"], ax=ax)

    ax.set(xlabel="Time (bins)", ylabel="Intraday Volume")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(title="", loc="lower center", fontsize=12, frameon=False)
    plt.tight_layout()

    return fig
