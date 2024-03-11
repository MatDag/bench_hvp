import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str,
                    default='../outputs/bench_hvp_time_jax.parquet')

filename = parser.parse_args().file

LEGEND_INSIDE = True
LEGEND_RATIO = 0.1
DEFAULT_WIDTH = 8
DEFAULT_HEIGHT = 4

fontsize = 12

STYLES = dict(
    grad=dict(label='Gradient', color='#5778a4'),
    hvp_forward_over_reverse=dict(label='HVP forward-over-reverse',
                                  color='#e49444'),
    hvp_reverse_over_forward=dict(label='HVP reverse-over-forward',
                                  color='#e7ca60'),
    hvp_reverse_over_reverse=dict(label='HVP reverse-over-reverse',
                                  color='#d1615d'),
)


MODELS = dict(
    resnet34=dict(label="ResNet34", color="#ffe6e6", ord=1),
    bert=dict(label="BERT", color="#defcce", ord=2),
    vit=dict(label="ViT", color="#fcf8c1", ord=3),
)

FUN = "hvp_forward_over_reverse"

mpl.rcParams.update({
    'font.size': fontsize,
    'legend.fontsize': fontsize,
    'axes.labelsize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize
})

df = (
        pd.read_parquet(filename)
        .reset_index().query('model in @MODELS.keys()')
)
models = df['model'].unique()
funs = df['fun'].unique()
for model in models:
    df.loc[df['model'] == model, 'ord'] = MODELS[model]['ord']

frameworks = df['framework'].unique()
frameworks = " / ".join(frameworks)

fig = plt.figure(
    figsize=(DEFAULT_WIDTH / 1.8 * 2,
             DEFAULT_HEIGHT*(1 + LEGEND_RATIO))
)
gs = plt.GridSpec(
    2, 2, height_ratios=[LEGEND_RATIO, 1],
    hspace=.1, bottom=.1, top=.95
)


ax = fig.add_subplot(gs[1, 0])
lines = []

width = .20
space = .05
n_funs = len(df['fun'].unique())
mutliplier = np.arange(n_funs)
x = np.arange(len(models)) * (1 + space)

max_batch_sizes = df.groupby('model').max()['batch_size']


def filtre_max_batch_size(df, max_batch_sizes):
    new_df = pd.DataFrame()
    for model in df['model'].unique():
        batch_size = max_batch_sizes[model]
        new_df = pd.concat([
            new_df,
            df.query(f"model == @model & batch_size == {batch_size}")
        ])
    return new_df


df_batch_size_64 = df.query("batch_size == 64")

for j, fun in enumerate(df['fun'].unique()):
    to_plot = (
        df_batch_size_64.query("fun == @fun")
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

ax.set_ylabel('Time [sec]')
ax.set_xticks(ticks=x + 1.5*width,
              labels=[MODELS[m]['label'] for m in models],
              fontsize=fontsize)

df = (
    df
    .groupby(['fun', 'model', 'batch_size'])
    .quantile([0.2, 0.5, 0.8], numeric_only=True)
    .reset_index(level='fun')
)

lines = []

model = 'resnet34'
ax = fig.add_subplot(gs[1, 1])
df_model = df.query("model == @model")
colors = iter([plt.cm.Set1(i) for i in range(4)])
for fun in funs:
    to_plot = df_model.query("fun == @fun")

    lines.append(
        ax.plot(
            to_plot.index.get_level_values(1).unique(),
            to_plot.iloc[to_plot.index.get_level_values(2) == .5]["time"],
            color=STYLES[fun]['color'],
            marker='o',
            label=STYLES[fun]['label'],
        )[0]
    )
    ax.fill_between(
        to_plot.index.get_level_values(1).unique(),
        to_plot.iloc[to_plot.index.get_level_values(2) == .2]["time"],
        to_plot.iloc[to_plot.index.get_level_values(2) == .8]["time"],
        color=STYLES[fun]['color'],
        alpha=.2,
    )

ax.set_xlabel('Batch size')
ax_legend = fig.add_subplot(gs[0, :])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=2, fontsize=fontsize)

plt.savefig(f'bench_hvp_time_{frameworks}.png', dpi=300)
