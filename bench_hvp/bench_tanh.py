import jax
import jax.numpy as jnp

import torch

import pandas as pd
from functools import partial

from memory_profiler import memory_usage

from time import perf_counter

import yaml
import submitit
import itertools
from rich import progress
from submitit.helpers import as_completed

from joblib import Memory
mem = Memory(location='__cache__')

N_REPS = 100
DIM_LIST = jnp.logspace(1, 5, 14, dtype=jnp.int32)
SLURM_CONFIG = 'config/slurm_cpu.yml'


@jax.jit
def f_jax(x):
    return jnp.sum(jnp.tanh(x)**2)


@torch.compile
def f_torch(x):
    return torch.sum(torch.tanh(x)**2)


@mem.cache
def run_one(fun_name, framework='jax', dim=1, n_reps=1):
    if framework == 'jax':
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (dim, ))
        fun = f_jax
        grad_fun = jax.jit(jax.grad(fun))
    elif framework == 'torch':
        gen = torch.Generator()
        gen.manual_seed(0)
        x = torch.randn(dim, generator=gen)
        fun = f_torch
        grad_fun = torch.func.grad(fun)

    if fun_name == "grad":
        grad_fun(x)  # First run for compilation
    else:
        v = grad_fun(x)  # First run to get a v and for compilation
        hvp_fun = fun_dict[fun_name]
        hvp_fun(x, v)  # First run for compilation
    times = []
    memories = []
    for _ in range(n_reps):
        if fun_name == "grad":
            start = perf_counter()
            jax.block_until_ready(grad_fun(x))
            time = perf_counter() - start
            times.append(time)
            memories.append(max(memory_usage((grad_fun, (x, )))))
        else:
            start = perf_counter()
            jax.block_until_ready(hvp_fun(x, v))
            time = perf_counter() - start
            times.append(time)
            memories.append(max(memory_usage((hvp_fun, (x, v)))))

    return dict(
        dim=float(dim),
        label=fun_name,
        time=times,
        memory=memories,
        rep=jnp.arange(n_reps),
    )


def run_bench(fun_list, dim_list, n_reps, slurm_config_path=SLURM_CONFIG):
    run = partial(run_one, n_reps=n_reps)

    with open(slurm_config_path, "r") as f:
        config = yaml.safe_load(f)

    executor = submitit.AutoExecutor("bench_hvp")
    executor.update_parameters(**config)

    with executor.batch():
        jobs = [
            executor.submit(run,
                            fun_name,
                            dim)
            for fun_name, dim in itertools.product(fun_list, dim_list)
        ]
    print(f"First job ID: {jobs[0].job_id}")

    for t in progress.track(as_completed(jobs), total=len(jobs)):
        exc = t.exception()
        if exc is not None:
            for tt in jobs:
                tt.cancel()
            raise exc

    results = [t.result() for t in jobs]
    return pd.concat([pd.DataFrame(res) for res in results])


@jax.jit
def hvp_naive_jax(x, v):
    """
    Returns the Hessian-vector product by computing the full Hessian matrix
    and then multiplying by the tangent vector v.
    """

    return jnp.dot(jax.hessian(f_jax)(x), v)


def hvp_naive_torch(x, v):
    """
    Returns the Hessian-vector product by computing the full Hessian matrix
    and then multiplying by the tangent vector v.
    """

    return torch.dot(torch.func.hessian(f_torch)(x), v)


@jax.jit
def hvp_forward_over_reverse_jax(x, v):
    """
    Returns the Hessian-vector product by forward-over-reverse propagation.
    """
    return jax.jvp(jax.grad(f_jax), (x, ), (v, ))[1]


def hvp_forward_over_reverse_torch(x, v):
    """
    Returns the Hessian-vector product by forward-over-reverse propagation.
    """
    return torch.func.jvp(torch.func.grad(f_torch), (x, ), (v, ))[1]


@jax.jit
def hvp_reverse_over_forward_jax(x, v):
    """
    Returns the Hessian-vector product by reverse-over-forward propagation.
    """
    def jvp_fun(x, v):
        return jax.jvp(f_jax, (x, ), (v, ))[1]

    return jax.grad(jvp_fun)(x, v)


def hvp_reverse_over_forward_torch(x, v):
    """
    Returns the Hessian-vector product by reverse-over-forward propagation.
    """
    def jvp_fun(x, v):
        return torch.func.jvp(f_torch, (x, ), (v, ))[1]

    return jax.grad(jvp_fun)(x, v)


@jax.jit
def hvp_reverse_over_reverse_jax(x, v):
    """
    Returns the Hessian-vector product by reverse-over-reverse propagation.
    """
    return jax.grad(lambda x: jnp.dot(jax.grad(f_jax)(x), v))(x)


def hvp_reverse_over_reverse_torch(x, v):
    """
    Returns the Hessian-vector product by reverse-over-reverse propagation.
    """
    return torch.func.grad(
        lambda x: torch.dot(torch.func.grad(f_torch)(x), v)
    )(x)


if __name__ == '__main__':
    fun_dict = dict(
        # grad=None,
        hvp_naive_jax=hvp_naive_jax,
        hvp_forward_over_reverse_jax=hvp_forward_over_reverse_jax,
        hvp_reverse_over_forward_jax=hvp_reverse_over_forward_jax,
        hvp_reverse_over_reverse_jax=hvp_reverse_over_reverse_jax,
        hvp_naive_torch=hvp_naive_torch,
        hvp_forward_over_reverse_torch=hvp_forward_over_reverse_torch,
        hvp_reverse_over_forward_torch=hvp_reverse_over_forward_torch,
        hvp_reverse_over_reverse_torch=hvp_reverse_over_reverse_torch,
    )
    fun_list = fun_dict.keys()
    df = run_bench(fun_list, DIM_LIST, n_reps=N_REPS,
                   slurm_config_path=SLURM_CONFIG)
    df.to_parquet('../outputs/bench_tanh.parquet')
