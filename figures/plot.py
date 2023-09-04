import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

LEGEND_RATIO = 0.1
DEFAULT_WIDTH = 5
DEFAULT_DOUBLE_WIDTH = 6.75
DEFAULT_HEIGHT = 3

STYLES = {
    'Gradient': '#5778a4',
    'HVP forward-over-reverse': '#e49444',
    'HVP reverse-over-forward': '#e7ca60',
    'HVP reverse-over-reverse': '#d1615d',
}

mpl.rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'small',
    'axes.labelsize': 'small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
})

df = pd.read_parquet('../outputs/bench_hvp.parquet')

fig = plt.figure(
    figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT)
)
gs = plt.GridSpec(1, 2, width_ratios=[.6, .4], hspace=0, bottom=.1)
ax = fig.add_subplot(gs[0, 0])

lines = []

for label in df['label'].unique():
    to_plot = (
        df.query("label == @label")
        .groupby(['depth'])
        .quantile([0.2, 0.5, 0.8], numeric_only=True)
        .unstack()
    )
    lines.append(
        ax.plot(
            to_plot.index,
            to_plot.loc[:, ("time", 0.5)],
            color=STYLES[label],
            label=label
        )[0]
    )
    ax.fill_between(
        to_plot.index,
        to_plot.loc[:, ("time", 0.2)],
        to_plot.loc[:, ("time", 0.8)],
        alpha=0.3,
        color=STYLES[label]
    )
ax.set_xlabel('Depth')
ax.set_ylabel('Time [sec]')

ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=1)

plt.savefig('bench_hvp_test.pdf', dpi=300)
