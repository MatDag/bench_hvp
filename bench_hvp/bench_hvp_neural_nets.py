import jax
import optax
import jax.numpy as jnp
from flax.training import common_utils

import pandas as pd
from functools import partial

from time import perf_counter

import yaml
import submitit
import itertools
from rich import progress
from submitit.helpers import as_completed

from transformers import FlaxViTForImageClassification
from transformers import FlaxResNetForImageClassification
from transformers import FlaxBertForSequenceClassification

import utils

from joblib import Memory
mem = Memory(location='__cache__')

N_REPS = 100
BATCH_SIZE_LIST = [16, 32, 64, 128]
MODEL_DICT = dict(
    resnet18=dict(module=FlaxResNetForImageClassification,
                  model="microsoft/resnet-18", num_classes=1000),
    resnet34=dict(module=FlaxResNetForImageClassification,
                  model="microsoft/resnet-34", num_classes=1000),
    resnet50=dict(module=FlaxResNetForImageClassification,
                  model="microsoft/resnet-50", num_classes=1000),
    resnet101=dict(module=FlaxResNetForImageClassification,
                   model="microsoft/resnet-101", num_classes=1000),
    resnet152=dict(module=FlaxResNetForImageClassification,
                   model="microsoft/resnet-152", num_classes=1000),
    vit=dict(module=FlaxViTForImageClassification,
             model="google/vit-base-patch16-224", num_classes=1000),
    bert=dict(module=FlaxBertForSequenceClassification,
              model="bert-base-uncased", num_classes=2),
)
SLURM_CONFIG = 'config/slurm.yml'


def cross_entropy_loss(logits, labels, num_classes=1000):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits,
                                           labels=one_hot_labels)
    return jnp.mean(xentropy)


def loss_fn(params, model, batch, num_classes=1000):
    """loss function used for training."""
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    if 'images' in inputs.keys():
        logits = model._module.apply(params, inputs['images']).logits
    else:
        logits = model._module.apply(params, **inputs).logits
    loss = cross_entropy_loss(logits, batch['labels'], num_classes=num_classes)
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x ** 2)
                    for x in weight_penalty_params
                    if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss


@mem.cache
def run_one(fun_name, model_name, batch_size=16, n_reps=1):
    key = jax.random.PRNGKey(0)
    num_classes = MODEL_DICT[model_name]['num_classes']
    if model_name != "bert":
        key, subkey = jax.random.split(key)
        batch = {
            'images': jax.random.normal(key, (batch_size, 224, 224, 3)),
            'labels': jax.random.randint(subkey, (batch_size,), 0, num_classes)
        }
    else:
        keys = jax.random.split(key, 4)
        batch = {
            'input_ids': jax.random.randint(keys[0], (batch_size, 128),
                                            0, 10000),
            'attention_mask': jax.random.randint(keys[1], (batch_size, 128),
                                                 0, 2),
            'token_type_ids': jax.random.randint(keys[2], (batch_size, 128),
                                                 0, 2),
            'position_ids': None,
            'head_mask': None,
            'labels': jax.random.randint(keys[3], (batch_size,),
                                         0, num_classes)
        }
    key, subkey = jax.random.split(key)
    model = MODEL_DICT[model_name]['module'].from_pretrained(
        MODEL_DICT[model_name]['model']
    )
    # init = model.init(key, batch['images'], train=True)
    params = model.params
    if "params" not in params.keys():
        params = {"params": params}
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, num_classes=num_classes)
    )

    if fun_name == "grad":
        grad_fun(params)  # First run for compilation
    else:
        v = grad_fun(params)  # First run to get a v and for compilation
        hvp_fun = fun_dict[fun_name]['fun'](model, batch,
                                            num_classes=num_classes)
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
        batch_size=batch_size,
        rep=jnp.arange(n_reps),
    )


def run_bench(fun_list, model_list, n_reps, batch_size_list, num_classes=1000,
              slurm_config_path=None):
    run = partial(run_one, n_reps=n_reps)

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


def hvp_forward_over_reverse(model, batch, num_classes=1000):
    """
    Returns the Hessian-vector product operator that uses forward-over-reverse
    propagation.
    """
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, num_classes=num_classes)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.jvp(grad_fun, (x, ), (v, ))[1]
    )
    return hvp_fun


def hvp_reverse_over_forward(model, batch, num_classes=1000):
    """
    Returns the Hessian-vector product operator that uses reverse-over-forward
    propagation.
    """
    jvp_fun = jax.jit(
        lambda x, v: jax.jvp(
            lambda y: loss_fn(y, model, batch, num_classes=num_classes),
            (x, ), (v, )
        )[1]
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.grad(jvp_fun)(x, v)
    )

    return hvp_fun


def hvp_reverse_over_reverse(model, batch, num_classes=1000):
    """
    Returns the Hessian-vector product operator that uses reverse-over-reverse
    propagation.
    """

    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, num_classes=num_classes)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.grad(lambda x: utils.tree_dot(grad_fun(x), v))(x)
    )
    return hvp_fun


if __name__ == '__main__':
    fun_dict = dict(
        grad=dict(fun=None, label="Gradient"),
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
    df.to_parquet('../outputs/bench_hvp_neural_nets.parquet')
