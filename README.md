# Benchmark Hessian-vector product

This Python repo hosts the code of the blogpost about Hessian-vector products.

## Installation


```bash
$ git clone https://github.com/MatDag/bench_hvp.git
```

## Reproducing the figures 

### Time complexity

To reproduce the computational time comparison in Jax or in PyTorch, run the following command where `XXX` has to be replaced by `jax` or `torch` accordingly

```bash
$ conda activate bench_hvp_XXX
$ cd bench_hvp
$ python bench_hvp_time.py -f XXX -n 100
```

The raw results are stored in the folder `outputs`. Then, the figure can be plotted 
```bash
$ cd ../figures
$ python plot_bench_time.py -f ../outputs/bench_hvp_time_XXX.parquet
```

The figures are stored in the folder `figures`.
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_time_jax.pdf)
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_time_torch.pdf)

```bash
$ conda activate bench_hvp_torch
$ python bench_hvp_jax -f torch -n 90
```
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_time_torch.pdf)