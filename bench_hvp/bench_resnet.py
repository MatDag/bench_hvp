import jax
import optax
import jax.numpy as jnp
from models import resnet_flax
from flax.training import common_utils

import torch
from torchvision.models import resnet as resnet_torch
from functorch.experimental import replace_all_batch_norm_modules_

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
N_REPS = 10
BATCH_SIZE_LIST = [16]
MODEL_DICT = dict(
    resnet18_flax=dict(model=resnet_flax.ResNet18, framework='jax'),
    resnet34_flax=dict(model=resnet_flax.ResNet34, framework='jax'),
    resnet50_flax=dict(model=resnet_flax.ResNet50, framework='jax'),
    resnet101_flax=dict(model=resnet_flax.ResNet101, framework='jax'),
    resnet152_flax=dict(model=resnet_flax.ResNet152, framework='jax'),
    resnet18_torch=dict(model=resnet_torch.resnet18, framework='torch'),
    resnet34_torch=dict(model=resnet_torch.resnet34, framework='torch'),
    resnet50_torch=dict(model=resnet_torch.resnet50, framework='torch'),
    resnet101_torch=dict(model=resnet_torch.resnet101, framework='torch'),
    resnet152_torch=dict(model=resnet_torch.resnet152, framework='torch'),
)
SLURM_CONFIG = 'config/slurm_margaret.yml'


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    xentropy = optax.softmax_cross_entropy(logits=logits,
                                           labels=one_hot_labels)
    return jnp.mean(xentropy)


def loss_fn_jax(params, model, batch, batch_stats):
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


def loss_fn_torch(params, model, batch):
    """loss function used for training."""
    logits = torch.func.functional_call(model, params, (batch['images'], ))
    loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
    weight_decay = 0.0001
    weight_l2 = sum(p.norm()**2 for p in params.values() if p.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    return loss + weight_penalty


@mem.cache
def run_one(fun_name, model_name, framework='jax', batch_size=16, n_reps=1,
            num_classes=NUM_CLASSES):
    if framework == 'jax':
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        batch = {
            'images': jax.random.normal(key, (batch_size, 128, 128, 3)),
            'labels': jax.random.randint(subkey, (batch_size,), 0, num_classes)
        }

        key, subkey = jax.random.split(key)
        model = MODEL_DICT[model_name]['model'](num_classes=num_classes)
        init = model.init(key, batch['images'], train=True)
        params, batch_stats = init['params'], init['batch_stats']
        grad_fun = jax.jit(
            lambda x: jax.grad(loss_fn_jax)(x, model, batch, batch_stats)
        )
    elif framework == 'torch':
        use_gpu = torch.cuda.is_available()
        gen = torch.Generator().manual_seed(0)
        batch = {
            'images': torch.randn(batch_size, 3, 128, 128, generator=gen),
            'labels': torch.randint(0, num_classes, (batch_size,),
                                    generator=gen)
        }
        if use_gpu:
            torch.cuda.empty_cache()
        model = MODEL_DICT[model_name]['model'](num_classes=num_classes)
        replace_all_batch_norm_modules_(model)
        if use_gpu:
            batch = {k: v.cuda() for k, v in batch.items()}
            model = model.cuda()
        params = dict(model.named_parameters())

        def grad_fun(x):
            return torch.func.grad(loss_fn_torch)(x, model, batch)
        batch_stats = None

    if fun_name == "grad":
        start = perf_counter()
        grad_fun(params)  # First run for compilation
    else:
        v = grad_fun(params)  # First run to get a v and for compilation
        hvp_fun = fun_dict[fun_name]['fun'](model, batch, batch_stats,
                                            framework=framework)
        hvp_fun(params, v)  # First run for compilation
    times = []
    for _ in range(n_reps):
        if fun_name == "grad":
            if framework == 'jax':
                start = perf_counter()
                jax.block_until_ready(grad_fun(params))
                time = perf_counter() - start
            elif framework == 'torch':
                start = perf_counter()
                grad_fun(params)
                time = perf_counter() - start
            times.append(time)
        else:
            if framework == 'jax':
                start = perf_counter()
                jax.block_until_ready(hvp_fun(params, v))
                time = perf_counter() - start
                start = perf_counter()
                jax.block_until_ready(grad_fun(params))
                grad_time = perf_counter() - start
            elif framework == 'torch':
                start = perf_counter()
                hvp_fun(params, v)
                time = perf_counter() - start
                start = perf_counter()
                grad_fun(params)
                grad_time = perf_counter() - start
            times.append(time - grad_time)

    return dict(
        model=model_name,
        label=fun_name,
        time=times,
        batch_size=batch_size,
        framework=framework,
        rep=jnp.arange(n_reps),
    )


def run_bench(fun_list, model_list, n_reps, batch_size_list, num_classes=1000,
              slurm_config_path=None, framework='jax'):
    run = partial(run_one, num_classes=num_classes, n_reps=n_reps,
                  framework=framework)

    with open(slurm_config_path, "r") as f:
        config = yaml.safe_load(f)

    executor = submitit.AutoExecutor("bench_hvp")
    executor.update_parameters(**config)
    skip = [(128, 'resnet50_torch'),
            (64, 'resnet101_torch'), (128, 'resnet101_torch'),
            (64, 'resnet152_torch'), (128, 'resnet152_torch')]

    with executor.batch():
        jobs = [
            executor.submit(run,
                            fun_name,
                            model_name,
                            batch_size)
            for fun_name, model_name, batch_size
            in itertools.product(fun_list, model_list, batch_size_list)
            if (batch_size, model_name) not in skip
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


def hvp_forward_over_reverse(model, batch, batch_stats=None, framework='jax'):
    """
    Returns the time taken to compute the Hessian-vector product by
    forward-over-reverse propagation.
    """
    if framework == 'jax':
        grad_fun = jax.jit(
            lambda x: jax.grad(loss_fn_jax)(x, model, batch, batch_stats)
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.jvp(grad_fun, (x, ), (v, ))[1]
        )
    elif framework == 'torch':
        def grad_fun(x):
            return torch.func.grad(loss_fn_torch)(x, model, batch)

        def hvp_fun(x, v):
            return torch.func.jvp(grad_fun, (x, ), (v, ))[1]
    return hvp_fun


def hvp_reverse_over_forward(model, batch, batch_stats=None, framework='jax'):
    """
    Returns the time taken to compute the
    Hessian-vector product by reverse-over-forward propagation.
    """
    if framework == 'jax':
        jvp_fun = jax.jit(
            lambda x, v: jax.jvp(
                lambda y: loss_fn_jax(y, model, batch, batch_stats),
                (x, ), (v, )
            )[1]
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.grad(jvp_fun)(x, v)
        )
    elif framework == 'torch':
        def jvp_fun(x, v):
            return torch.func.jvp(
                lambda y: loss_fn_torch(y, model, batch),
                (x, ), (v, )
            )[1]

        def hvp_fun(x, v):
            return torch.func.grad(jvp_fun)(x, v)

    return hvp_fun


def hvp_reverse_over_reverse(model, batch, batch_stats=None, framework='jax'):
    """
    Returns the time taken to compute the Hessian-vector product by
    reverse-over-reverse propagation.
    """
    if framework == 'jax':
        grad_fun = jax.jit(
            lambda x: jax.grad(loss_fn_jax)(x, model, batch, batch_stats)
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.grad(lambda x: utils.tree_dot(grad_fun(x), v))(x)
        )
    elif framework == 'torch':
        def grad_fun(x):
            return torch.func.grad(loss_fn_torch)(x, model, batch)

        def hvp_fun(x, v):
            return torch.func.grad(
                lambda x: sum(
                    torch.dot(a.ravel(), b.ravel())
                    for a, b in zip(grad_fun(x).values(), v.values())
                )
            )(x)
    return hvp_fun


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', '-f', type=str, default='jax',
                        choices=['jax', 'torch'])
    framework = parser.parse_args().framework
    fun_dict = dict(
        grad=dict(fun=None, label="Gradient"),
        # hvp_naive=dict(fun=hvp_naive, label="HVP naive"),
        hvp_forward_over_reverse=dict(
            fun=hvp_forward_over_reverse, label="HVP forward-over-reverse"),
        hvp_reverse_over_forward=dict(
            fun=hvp_reverse_over_forward, label="HVP reverse-over-forward"),
        hvp_reverse_over_reverse=dict(
            fun=hvp_reverse_over_reverse, label="HVP reverse-over-reverse"),
    )
    model_list = [k for k in MODEL_DICT.keys()
                  if MODEL_DICT[k]['framework'] == framework]
    fun_list = fun_dict.keys()
    df = run_bench(fun_list, model_list, n_reps=N_REPS,
                   batch_size_list=BATCH_SIZE_LIST,
                   slurm_config_path=SLURM_CONFIG,
                   framework=framework)
    df.to_parquet(f'../outputs/bench_resnet_{framework}.parquet')
