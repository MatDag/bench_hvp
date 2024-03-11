# Benchmark Hessian-vector product

```bash
$ conda activate bench_hvp_jax
$ python bench_hvp_jax -f jax -n 90
```
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_time_jax.pdf)

```bash
$ conda activate bench_hvp_torch
$ python bench_hvp_jax -f torch -n 90
```
![alt text](https://github.com/MatDag/bench_hvp/blob/main/figures/bench_hvp_time_torch.pdf)