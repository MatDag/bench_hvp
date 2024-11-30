# Benchmark Hessian-vector product

This Python repo hosts the code of the blog post "[How to compute Hessian-vector products?](https://iclr-blogposts.github.io/2024/blog/bench-hvp/)" published in the blog post track of ICLR 2024.
This blog post [received a highlight](https://blog.iclr.cc/2024/04/02/blogposts-track-iclr-2023-announcing-accepted-blogposts/).

## Installation

Start by cloning the repo:
```bash
$ git clone https://github.com/MatDag/bench_hvp.git
$ cd bench_hvp
```
We advise working in two separate environments: one environment for the experiments in Jax and another for the PyTorch experiments.

### Jax environment
To set up the Jax environment, run:
```bash
$ conda create -n bench_hvp_jax python=3.11
$ conda activate bench_hvp_jax
$ pip install -e .
$ pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
In the blogpost, the experiments were done with version `0.4.21` of Jax.

### PyTorch environment
To set up the Jax environment, run:
```bash
$ conda create -n bench_hvp_torch python=3.11
$ conda activate bench_hvp_torch
$ pip install -e .
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
In the blogpost, the experiments were done with version `2.1.1.` of PyTorch.


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
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_time_jax.png)
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_time_torch.png)

### Memory complexity

To reproduce the memory footprint experiment, you can execute the following bash script

```bash
$ cd bench_hvp
$ bash bench_hvp_memory.sh
```

The raw results are stored in the folder `outputs`. Then, the figure can be plotted 
```bash
$ cd ../figures
$ python plot_bench_memory.py -f ../outputs/bench_hvp_memory_XXX.parquet
```
where `XXX` can be replaced by `jax`, `jax_without_jit` or `torch`.
The figures are stored in the folder `figures`.

![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_memory_jax.png)
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_memory_jax_without_jit.png)
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_memory_torch.png)
