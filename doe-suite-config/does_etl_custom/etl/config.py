
import matplotlib.pyplot as plt

def setup_plt():
    fig_size = [8.041760066417601, 3.8838667729342697]
    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.size': 12,
        'figure.figsize': fig_size,
        'font.family': 'Times New Roman',
        'lines.markersize': 8
    }

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3
