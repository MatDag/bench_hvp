import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BATCH_SIZE = 16

LEGEND_INSIDE = True
LEGEND_RATIO = 0.1
DEFAULT_WIDTH = 8
DEFAULT_HEIGHT = 5

fontsize = 7

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
    resnet50=dict(label="ResNet50", color="#ffe6e6"),
    vit=dict(label="ViT", color="#fcf8c1"),
    bert=dict(label="BERT", color="#defcce"),
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
        pd.read_parquet('../outputs/bench_hvp_memory_jax.parquet')
        .reset_index()
)


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
    to_plot = df.query("fun == @fun & batch_size == @BATCH_SIZE")

    lines.append(
        ax.bar(
            x + mutliplier[j] * width,
            to_plot.loc[:, "memory"],
            width=width,
            yerr=[
                to_plot.loc[:, "memory"]
                - to_plot.loc[:, "memory"],
                to_plot.loc[:, "memory"]-to_plot.loc[:, "memory"]
            ],
            color=STYLES[fun]['color'],
            label=STYLES[fun]['label'],
        )
    )

ax.set_ylabel('Memory (MB)')
ax.set_xticks(ticks=x + width,
              labels=[MODELS[m]['label']
                      for m in to_plot.loc[:, "model"].unique()],
              fontsize=fontsize)

ax_legend = fig.add_subplot(gs[0, 0])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=1, fontsize=fontsize)

ax = fig.add_subplot(gs[1, 1])

to_plot = df.query('fun == @FUN')

lines = []

colors = iter([plt.cm.Set1(i) for i in range(3)])

for j, model in enumerate(MODELS):

    lines.append(
        ax.plot(
            to_plot.query("model == @model").loc[:, "batch_size"],
            to_plot.query("model == @model").loc[:, "memory"],
            color=next(colors),
            marker='o',
            label=MODELS[model]['label'],
        )[0]
    )

ax.set_xlabel('Batch size')

ax.set_xticks(ticks=to_plot.loc[:, "batch_size"].unique())

ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=1, fontsize=fontsize)
# ax.set_ylabel('Memory (MB)')

plt.savefig('bench_hvp_memory_jax.png', dpi=300)
