import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_parquet('../outputs/bench_hvp64.parquet')

plt.figure(figsize=(6, 4))

plt.plot(df['depth'], df["grad_mean_time"], "r", label="grad")
plt.fill_between(df["depth"], df["grad_mean_time"]+df["grad_std_time"],
                 df["grad_mean_time"]-df["grad_std_time"], alpha=.4, color="r")

plt.plot(df['depth'], df["hvp_mean_time"], "b", label="hvp and grad")
plt.fill_between(df["depth"], df["hvp_mean_time"]+df["hvp_std_time"],
                 df["hvp_mean_time"]-df["hvp_std_time"], alpha=.4, color="b")

plt.plot(df['depth'], df["hvp_mean_time"] - df["grad_mean_time"], "g",
         label="hvp")
plt.xlabel("Depth")
plt.ylabel("Time [sec]")
plt.legend()

plt.savefig('bench_hvp64.pdf', dpi=300)
