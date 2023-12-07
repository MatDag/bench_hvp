import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BATCH_SIZE = 64

LEGEND_INSIDE = True
LEGEND_RATIO = 0.1
DEFAULT_WIDTH = 8
DEFAULT_HEIGHT = 6

fontsize = 7

STYLES = dict(
    grad=dict(label='Gradient', color='#5778a4'),
    hvp_forward_over_reverse=dict(label='HVP forward-over-reverse',
                                  color='#e49444'),
    hvp_reverse_over_forward=dict(label='HVP reverse-over-forward',
                                  color='#e7ca60'),
    hvp_reverse_over_reverse=dict(label='HVP reverse-over-reverse',
                                  color='#d1615d'),
    # torch_vhp=dict(label='Torch vhp', color='red'),
    # torch_hvp=dict(label='Torch HVP', color='orange'),
)


MODELS = dict(
    resnet34=dict(label="ResNet34", color="#ffe6e6", ord=1),
    bert=dict(label="BERT", color="#defcce", ord=2),
    vit=dict(label="ViT", color="#fcf8c1", ord=3),
)

FUN = "hvp_forward_over_reverse"

mpl.rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'small',
    'axes.labelsize': 'small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
})

df = (
        pd.read_parquet('../outputs/bench_hvp_time_jax4.parquet')
        .reset_index().query('model != "resnet50"')
)

for model in MODELS:
    df.loc[df['model'] == model, 'ord'] = MODELS[model]['ord']

fig = plt.figure(
    figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT*(1 + LEGEND_RATIO))
)
gs = plt.GridSpec(2, 2, width_ratios=[1, 1],
                  height_ratios=[LEGEND_RATIO, 1],
                  hspace=.1, bottom=.1, top=.95)


ax = fig.add_subplot(gs[1, 0])
lines = []

width = .20
space = .05
n_funs = len(df['fun'].unique())
mutliplier = np.arange(n_funs)
x = np.arange(len(MODELS)) * (1 + space)

for j, fun in enumerate(df['fun'].unique()):
    # import ipdb; ipdb.set_trace()
    to_plot = (
        df.query("fun == @fun & batch_size == @BATCH_SIZE")
        .groupby(['model'])
        .quantile([0.2, 0.5, 0.8], numeric_only=True)
        .unstack().sort_values(by=('ord', 0.5))
    )

    lines.append(
        ax.bar(
            x + mutliplier[j] * width,
            to_plot.loc[:, ("time", 0.5)],
            width=width,
            yerr=[
                to_plot.loc[:, ("time", 0.5)]
                - to_plot.loc[:, ("time", 0.2)],
                to_plot.loc[:, ("time", 0.8)]-to_plot.loc[:, ("time", 0.5)]
            ],
            color=STYLES[fun]['color'],
            label=STYLES[fun]['label'],
        )
    )

ax.set_title(f'Batch size: {BATCH_SIZE}')
ax.set_ylabel('Time [sec]')
ax.set_xticks(ticks=x + 2*width,
              labels=[MODELS[m]['label'] for m in MODELS],
              fontsize=fontsize)

ax_legend = fig.add_subplot(gs[0, 0])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=1, fontsize=fontsize)

ax = fig.add_subplot(gs[1, 1], sharey=ax)

df_fun = (
            df.query('model == "resnet34"')
            .groupby(['fun', 'batch_size'])
            .quantile([0.2, 0.5, 0.8], numeric_only=True)
)

lines = []

colors = iter([plt.cm.Set1(i) for i in range(4)])
for j, fun in enumerate(STYLES):
    to_plot = df_fun.query("fun == @fun")
    color = next(colors)

    lines.append(
        ax.plot(
            to_plot.index.get_level_values(1).unique(),
            to_plot.iloc[to_plot.index.get_level_values(2) == .5]["time"],
            color=color,
            marker='o',
            label=STYLES[fun]['label'],
        )[0]
    )

    ax.fill_between(
        to_plot.index.get_level_values(1).unique(),
        to_plot.iloc[to_plot.index.get_level_values(2) == .2]["time"],
        to_plot.iloc[to_plot.index.get_level_values(2) == .8]["time"],
        color=color,
        alpha=.2,
    )

ax.set_xlabel('Batch size')
ax.set_title(r'resnet34, time vs batch size')
# ax.set_xticks(ticks=to_plot["batch_size"].unique())
ax.set_ylabel('Time [sec]')
ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=1, fontsize=fontsize)

fig.suptitle('Jax')

plt.savefig('bench_hvp_time_jax4.png', dpi=300)
