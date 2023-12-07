import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BATCH_SIZE = 128

LEGEND_RATIO = 0.1
DEFAULT_WIDTH = 9
DEFAULT_DOUBLE_WIDTH = 6.75
DEFAULT_HEIGHT = 5

STYLES = dict(
    # grad=dict(label='Gradient', color='#5778a4'),
    hvp_forward_over_reverse=dict(label='HVP forward-over-reverse',
                                  color='#e49444'),
    hvp_reverse_over_forward=dict(label='HVP reverse-over-forward',
                                  color='#e7ca60'),
    hvp_reverse_over_reverse=dict(label='HVP reverse-over-reverse',
                                  color='#d1615d'),
    hvp_naive=dict(label='HVP naive', color='#5778a4')
)


mpl.rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'small',
    'axes.labelsize': 'small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
})

df = pd.read_parquet('../outputs/bench_tanh2.parquet')


fig = plt.figure(
    figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT)
)
gs = plt.GridSpec(2, 2, height_ratios=[.15, 1], width_ratios=[.5, .5],
                  hspace=0.1, wspace=.25, bottom=.1)
ax = fig.add_subplot(gs[1, 0])

lines = []

for fun in STYLES:
    to_plot = (
        df.query("label == @fun")
        .groupby(['dim'])
        .quantile([0.35, 0.5, 0.65], numeric_only=True)
        .unstack()
    )
    lines.append(
        ax.plot(
            to_plot.index,
            to_plot.loc[:, ("time", 0.5)],
            color=STYLES[fun]['color'],
            label=STYLES[fun]['label'],
            marker='o'
        )[0]
    )
    ax.fill_between(
        to_plot.index,
        to_plot.loc[:, ("time", 0.35)],
        to_plot.loc[:, ("time", 0.65)],
        alpha=0.3,
        color=STYLES[fun]['color']
    )
ax.set_ylabel('Time [sec]')
ax.set_xlabel('Dimension')
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[1, 1])

lines = []

for fun in STYLES:
    to_plot = (
        df.query("label == @fun")
        .groupby(['dim'])
        .quantile([0.2, 0.5, 0.8], numeric_only=True)
        .unstack()
    )
    lines.append(
        ax.plot(
            to_plot.index,
            to_plot.loc[:, ("memory", 0.5)],
            color=STYLES[fun]['color'],
            label=STYLES[fun]['label'],
            marker='o'
        )[0]
    )
    ax.fill_between(
        to_plot.index,
        to_plot.loc[:, ("memory", 0.2)],
        to_plot.loc[:, ("memory", 0.8)],
        alpha=0.3,
        color=STYLES[fun]['color']
    )
ax.set_ylabel('Memory [MiB]')
ax.set_xlabel('Dimension')
ax.set_xscale('log')
ax.set_yscale('log')

ax_legend = fig.add_subplot(gs[0, :])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=2)

plt.savefig('bench_hvp_tanh.png', dpi=300)
