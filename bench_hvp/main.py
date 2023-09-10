import jax
import optax
import jax.numpy as jnp
from flax.training import common_utils
from models import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models import ResNet200

import pandas as pd
from functools import partial

from time import perf_counter

import yaml
import submitit
import itertools
from rich import progress
from submitit.helpers import as_completed

import utils

from joblib import Memory
mem = Memory(location='__cache__')

NUM_CLASSES = 1000
N_REPS = 100
BATCH_SIZE_LIST = [16, 32, 64, 128]
MODEL_DICT = dict(
    resnet18=ResNet18,
    resnet34=ResNet34,
    resnet50=ResNet50,
    resnet101=ResNet101,
    resnet152=ResNet152,
    resnet200=ResNet200,
)
SLURM_CONFIG = 'config/slurm_margaret.yml'


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
def run_one(fun_name, model_name, batch_size=16, n_reps=1,
            num_classes=NUM_CLASSES):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    batch = {
        'images': jax.random.normal(key, (batch_size, 128, 128, 3)),
        'labels': jax.random.randint(subkey, (batch_size,), 0, num_classes)
    }

    key, subkey = jax.random.split(key)
    model = MODEL_DICT[model_name](num_classes=num_classes)
    init = model.init(key, batch['images'], train=True)
    params, batch_stats = init['params'], init['batch_stats']
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, batch_stats)
    )

    if fun_name == "grad":
        grad_fun(params)  # First run for compilation
    else:
        v = grad_fun(params)  # First run to get a v and for compilation
        hvp_fun = fun_dict[fun_name]['fun'](model, batch, batch_stats)
        hvp_fun(params, v)  # First run for compilation
    times = []
    for _ in range(n_reps):
        if fun_name == "grad":
            start = perf_counter()
            jax.block_until_ready(grad_fun(params))
            time = perf_counter() - start
            times.append(time)
        else:
            start = perf_counter()
            jax.block_until_ready(hvp_fun(params, v))
            time = perf_counter() - start
            start = perf_counter()
            jax.block_until_ready(grad_fun(params))
            grad_time = perf_counter() - start
            times.append(time - grad_time)

    return dict(
        model=model_name,
        label=fun_name,
        time=times,
        rep=jnp.arange(n_reps),
    )


def run_bench(fun_list, model_list, n_reps, batch_size_list, num_classes=1000,
              slurm_config_path=None):
    run = partial(run_one, num_classes=num_classes, n_reps=n_reps)

    with open(slurm_config_path, "r") as f:
        config = yaml.safe_load(f)

    executor = submitit.AutoExecutor("bench_hvp")
    executor.update_parameters(**config)

    with executor.batch():
        jobs = [
            executor.submit(run,
                            fun_name,
                            model_name,
                            batch_size)
            for fun_name, model_name, batch_size
            in itertools.product(fun_list, model_list, batch_size_list)
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


def hvp_naive(params, model, batch, batch_stats):
    """
    Returns the time taken to compute the Hessian-vector product by computing
    the full Hessian matrix and then multiplying by the tangent vector v.
    """

    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, batch_stats)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.tree_map(jnp.dot,
                                  jax.hessian(loss_fn)(x, model,
                                                       batch, batch_stats),
                                  v)
    )

    v = grad_fun(params)  # First run to get a v and for compilation
    hvp_fun(params, v)  # First run for compilation

    start = perf_counter()
    jax.block_until_ready(hvp_fun(params, v))
    time = perf_counter() - start

    start = perf_counter()
    jax.block_until_ready(grad_fun(params))
    grad_time = perf_counter() - start
    return time - grad_time


def hvp_forward_over_reverse(model, batch, batch_stats):
    """
    Returns the time taken to compute the Hessian-vector product by
    forward-over-reverse propagation.
    """
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, batch_stats)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.jvp(grad_fun, (x, ), (v, ))[1]
    )
    return hvp_fun


def hvp_reverse_over_forward(model, batch, batch_stats):
    """
    Returns the time taken to compute the
    Hessian-vector product by reverse-over-forward propagation.
    """
    jvp_fun = jax.jit(
        lambda x, v: jax.jvp(
            lambda y: loss_fn(y, model, batch, batch_stats), (x, ), (v, )
        )[1]
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.grad(jvp_fun)(x, v)
    )

    return hvp_fun


def hvp_reverse_over_reverse(model, batch, batch_stats):
    """
    Returns the time taken to compute the Hessian-vector product by
    reverse-over-reverse propagation.
    """

    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, batch_stats)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.grad(lambda x: utils.tree_dot(grad_fun(x), v))(x)
    )
    return hvp_fun


if __name__ == '__main__':
    fun_dict = dict(
        grad=dict(fun=None, label="Gradient"),
        # hvp_naive=dict(fun=hvp_naive, label="HVP naive"),
        hvp_forward_over_reverse=dict(fun=hvp_forward_over_reverse,
                                      label="HVP forward-over-reverse"),
        hvp_reverse_over_forward=dict(fun=hvp_reverse_over_forward,
                                      label="HVP reverse-over-forward"),
        hvp_reverse_over_reverse=dict(fun=hvp_reverse_over_reverse,
                                      label="HVP reverse-over-reverse"),
    )
    model_list = MODEL_DICT.keys()
    fun_list = fun_dict.keys()
    df = run_bench(fun_list, model_list, n_reps=N_REPS,
                   batch_size_list=BATCH_SIZE_LIST,
                   slurm_config_path=SLURM_CONFIG)
    df.to_parquet('../outputs/bench_hvp.parquet')
