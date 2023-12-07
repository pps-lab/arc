
import matplotlib.pyplot as plt

def setup_plt(width=8., height=3.88):
    fig_size = [width, height]
    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 18,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 14,
        'ytick.labelsize': 16,
        'font.size': 12,
        'figure.figsize': fig_size,
        'font.family': 'Times New Roman',
        'lines.markersize': 8
    }

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3
