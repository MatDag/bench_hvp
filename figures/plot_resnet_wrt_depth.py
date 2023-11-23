import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BATCH_SIZE = 64

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

FRAMEWORK = dict(
    jax='Jax',
    torch='PyTorch'
)

MODELS = dict(
    resnet50=dict(label="ResNet50", ord=3),
    vit=dict(label="ViT", ord=6),
    bert=dict(label="BERT", ord=7),

)

mpl.rcParams.update({
    'font.size': 10,
    'legend.fontsize': 'small',
    'axes.labelsize': 'small',
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small'
})

df = pd.concat(
    [
        pd.read_parquet('../outputs/bench_hvp_neural_network_jax.parquet')
        .query("batch_size == @BATCH_SIZE"),
        pd.read_parquet('../outputs/bench_hvp_neural_network_torch.parquet')
        .query("batch_size == @BATCH_SIZE")
    ]
)

for model in MODELS:
    df.loc[df['model'] == model, 'ord'] = MODELS[model]['ord']

fig = plt.figure(
    figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT*(1 + LEGEND_RATIO))
)
gs = plt.GridSpec(2, 2, width_ratios=[1, 1],
                  height_ratios=[LEGEND_RATIO, 1],
                  hspace=.1, bottom=.1, top=.95)


for i, framework in enumerate(df['framework'].unique()):
    if i == 0:
        ax = fig.add_subplot(gs[1, i])
    else:
        ax = fig.add_subplot(gs[1, i], sharey=ax)
    lines = []

    width = .20
    space = .1
    n_funs = len(df['label'].unique())
    mutliplier = np.arange(n_funs)
    x = np.arange(len(MODELS)) * (1 + space)

    for j, fun in enumerate(df['label'].unique()):
        to_plot = (
            df.query("label == @fun & framework == @framework")
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

    if i == 0:
        ax.set_ylabel('Time [sec]')
    ax.set_yscale('log')
    ax.set_xticks(ticks=x + 2*width,
                  labels=[MODELS[m]['label'] for m in MODELS],
                  fontsize=fontsize)
    ax.set_title(FRAMEWORK[framework])

ax_legend = fig.add_subplot(gs[0, :])
ax_legend.set_axis_off()
ax_legend.legend(handles=lines, loc='center', ncol=2, fontsize=fontsize)

plt.savefig('bench_hvp_resnet_wrt_depth.png', dpi=300)
