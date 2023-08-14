import matplotlib.pyplot as plt
import pandas as pd

LEGEND_RATIO = 0.1
DEFAULT_WIDTH = 3.25
DEFAULT_DOUBLE_WIDTH = 6.75
DEFAULT_HEIGHT = 2.

METRIC_LIST = ["Time", "Mem"]

ORACLES = dict(
    hvp="HVP and grad",
    grad="Grad only",
    hvp_minus_grad="HVP only",
)

df = pd.read_parquet('../outputs/bench_hvp.parquet')

df['hvp_minus_grad_time'] = df['hvp_time'] - df['grad_time']
df['hvp_minus_grad_mem'] = df['hvp_mem'] - df['grad_mem']


fig = plt.figure(
    figsize=(DEFAULT_WIDTH, DEFAULT_HEIGHT * (1 + LEGEND_RATIO))
)
gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[.1, 1])
axes = [fig.add_subplot(gs[1, i]) for i in range(2)]

for ax, metric in zip(axes, METRIC_LIST):
    lines = []

    curves = (
        df.groupby(["depth"])
        .quantile([0.2, 0.5, 0.8], numeric_only=True)[
            ["hvp_" + metric.lower(),
             "grad_" + metric.lower(),
             "hvp_minus_grad_" + metric.lower()]
        ]
        .reset_index().set_index(['level_1']).sort_values('depth')
    )

    for oracle in ORACLES:
        y = f"{oracle}_{metric.lower()}"
        lines.append(
            curves.loc[0.5].plot(
                x="depth", y=y, ax=ax,
                label=ORACLES[oracle]
            )
        )
        ax.fill_between(
            curves.loc[0.5]["depth"],
            curves.loc[0.2][y],
            curves.loc[0.8][y],
            alpha=0.3
        )


plt.savefig('bench_hvp_test.pdf', dpi=300)
