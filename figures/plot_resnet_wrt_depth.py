import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BATCH_SIZE = 128

LEGEND_RATIO = 0.1
DEFAULT_WIDTH = 7
DEFAULT_DOUBLE_WIDTH = 6.75
DEFAULT_HEIGHT = 3

STYLES = dict(
    grad=dict(label='Gradient', color='#5778a4'),
    hvp_forward_over_reverse=dict(label='HVP forward-over-reverse',
                                  color='#e49444'),
    hvp_reverse_over_forward=dict(label='HVP reverse-over-forward',
                                  color='#e7ca60'),
    hvp_reverse_over_reverse=dict(label='HVP reverse-over-reverse',
                                  color='#d1615d')
)

MODELS = dict(
    resnet18=dict(label="ResNet18", ord=1),
    resnet34=dict(label="ResNet34", ord=2),
    resnet50=dict(label="ResNet50", ord=3),
    resnet101=dict(label="ResNet101", ord=4),
    resnet152=dict(label="ResNet152", ord=5),
    # resnet200=dict(label="ResNet200", ord=6),
)

mpl.rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'small',
    'axes.labelsize': 'small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
})

df = (
    pd.read_parquet('../outputs/bench_hvp.parquet')
    .query("batch_size == @BATCH_SIZE")
)

for model in MODELS:
    df.loc[df['model'] == model, 'ord'] = MODELS[model]['ord']

fig = plt.figure(
    figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT)
)
gs = plt.GridSpec(1, 2, width_ratios=[.6, .4], hspace=0, bottom=.1)
ax = fig.add_subplot(gs[0, 0])

lines = []

for fun in df['label'].unique():
    to_plot = (
        df.query("label == @fun")
        .groupby(['model'])
        .quantile([0.2, 0.5, 0.8], numeric_only=True)
        .unstack().sort_values(by=('ord', 0.5))
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
        to_plot.loc[:, ("time", 0.2)],
        to_plot.loc[:, ("time", 0.8)],
        alpha=0.3,
        color=STYLES[fun]['color']
    )
ax.set_ylabel('Time [sec]')
# ax.set_yscale('log')
ax.set_xticks(ticks=to_plot.index,
              labels=[MODELS[m]['label'] for m in MODELS])

ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=1)

plt.savefig('bench_hvp_resnet_wrt_depth.pdf', dpi=300)
