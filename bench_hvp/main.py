from models import ResNet, BottleneckResNetBlock

import jax
import optax
import jax.numpy as jnp
from flax.training import common_utils

import pandas as pd
from functools import partial

from time import perf_counter

from memory_profiler import memory_usage

import yaml
import submitit
import itertools
from rich import progress
from submitit.helpers import as_completed

from joblib import Memory
mem = Memory(location='__cache__')

NUM_CLASSES = 1000


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    xentropy = optax.softmax_cross_entropy(logits=logits,
                                           labels=one_hot_labels)
    return jnp.mean(xentropy)


def loss_fn(params, model, batch, batch_stats):
    """loss function used for training."""
    logits, _ = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        batch['images'],
        mutable=['batch_stats'])
    loss = cross_entropy_loss(logits, batch['labels'])
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x ** 2)
                    for x in weight_penalty_params
                    if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss


@mem.cache
def run_one(size, rep, batch_size=16, num_classes=1000):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    batch = {
        'images': jax.random.normal(key, (batch_size, 128, 128, 3)),
        'labels': jax.random.randint(subkey, (batch_size,), 0, num_classes)
    }

    key, subkey = jax.random.split(key)
    model = init_model(size)(num_classes=num_classes)
    init = model.init(key, batch['images'], train=True)
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, init['batch_stats'])
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.jvp(grad_fun, (x, ), (v, ))[1]
    )
    v = grad_fun(init['params'])  # First run to get a v and for compilation
    hvp_fun(init['params'], v)  # First run for compilation

    hvp_mem = max(memory_usage((hvp_fun, (init['params'], v))))
    start = perf_counter()
    jax.block_until_ready(hvp_fun(init['params'], v))
    hvp_time = perf_counter() - start

    grad_mem = max(memory_usage((grad_fun, (init['params'], ))))
    start = perf_counter()
    jax.block_until_ready(grad_fun(init['params']))
    grad_time = perf_counter() - start

    return dict(
        depth=float(2+3*(3+8+size+3)),
        hvp_time=hvp_time,
        hvp_mem=hvp_mem,
        grad_time=grad_time,
        grad_mem=grad_mem,
        rep=float(rep),
    )


def init_model(size):
    model = partial(ResNet, stage_sizes=[3, 8, size, 3],
                    block_cls=BottleneckResNetBlock)
    return model


def run_bench(sizes, reps, batch_size=16, num_classes=1000):
    run = partial(run_one, batch_size=batch_size,
                  num_classes=num_classes)

    with open('config/slurm_margaret.yml', "r") as f:
        config = yaml.safe_load(f)

    executor = submitit.AutoExecutor("bench_hvp")
    executor.update_parameters(**config)

    with executor.batch():
        jobs = [executor.submit(run, size, rep)
                for size, rep in itertools.product(sizes, reps)]
    print(f"First job ID: {jobs[0].job_id}")

    for t in progress.track(as_completed(jobs), total=len(jobs)):
        exc = t.exception()
        if exc is not None:
            for tt in jobs:
                tt.cancel()
            raise exc

    results = [t.result() for t in jobs]
    return pd.DataFrame(results)


if __name__ == '__main__':
    SIZES = jnp.arange(5, 50, 5)
    N_REPS = 10
    BATCH_SIZE = 16

    reps = jnp.arange(N_REPS)
    df = run_bench(SIZES, reps, batch_size=BATCH_SIZE)
    df.to_parquet('../outputs/bench_hvp.parquet')
