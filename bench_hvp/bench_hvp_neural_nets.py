import jax
import optax
import jax.numpy as jnp
from jax.lib import xla_bridge
from flax.training import common_utils

import torch
from functorch.experimental import replace_all_batch_norm_modules_

import pandas as pd
from functools import partial

from time import perf_counter

import yaml
import submitit
import itertools
from rich import progress
from submitit.helpers import as_completed

from transformers import ViTForImageClassification
from transformers import FlaxViTForImageClassification

from transformers import ResNetForImageClassification
from transformers import FlaxResNetForImageClassification

from transformers import BertForSequenceClassification
from transformers import FlaxBertForSequenceClassification

import utils

from joblib import Memory
mem = Memory(location='__cache__')

N_REPS = 100
BATCH_SIZE_LIST = [16, 32, 64, 128]
MODEL_DICT = dict(
    resnet18_flax=dict(module=FlaxResNetForImageClassification,
                       model="microsoft/resnet-18", framework="jax",
                       num_classes=1000),
    resnet34_flax=dict(module=FlaxResNetForImageClassification,
                       model="microsoft/resnet-34", framework="jax",
                       num_classes=1000),
    resnet50_flax=dict(module=FlaxResNetForImageClassification,
                       model="microsoft/resnet-50", framework="jax",
                       num_classes=1000),
    resnet101_flax=dict(module=FlaxResNetForImageClassification,
                        model="microsoft/resnet-101", framework="jax",
                        num_classes=1000),
    resnet152_flax=dict(module=FlaxResNetForImageClassification,
                        model="microsoft/resnet-152", framework="jax",
                        num_classes=1000),
    vit_flax=dict(module=FlaxViTForImageClassification,
                  model="google/vit-base-patch16-224", framework="jax",
                  num_classes=1000),
    bert_flax=dict(module=FlaxBertForSequenceClassification,
                   model="bert-base-uncased", framework="jax", num_classes=2),
    resnet18_torch=dict(module=ResNetForImageClassification,
                        model="microsoft/resnet-18", framework="torch",
                        num_classes=1000),
    resnet34_torch=dict(module=ResNetForImageClassification,
                        model="microsoft/resnet-34", framework="torch",
                        num_classes=1000),
    resnet50_torch=dict(module=ResNetForImageClassification,
                        model="microsoft/resnet-50", framework="torch",
                        num_classes=1000),
    resnet101_torch=dict(module=ResNetForImageClassification,
                         model="microsoft/resnet-101", framework="torch",
                         num_classes=1000),
    resnet152_torch=dict(module=ResNetForImageClassification,
                         model="microsoft/resnet-152", framework="torch",
                         num_classes=1000),
    vit_torch=dict(module=ViTForImageClassification,
                   model="google/vit-base-patch16-224", framework="torch",
                   num_classes=1000),
    bert_torch=dict(module=BertForSequenceClassification,
                    model="bert-base-uncased", framework="torch",
                    num_classes=2),
)
SLURM_CONFIG = 'config/slurm.yml'


def cross_entropy_loss(logits, labels, num_classes=1000):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits,
                                           labels=one_hot_labels)
    return jnp.mean(xentropy)


def loss_fn_jax(params, model, batch, num_classes=1000):
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


def loss_fn_torch(params, model, batch):
    """loss function used for training."""
    if 'images' in batch.keys():
        logits = torch.func.functional_call(model, params,
                                            (batch['images'], )).logits
    else:
        logits = torch.func.functional_call(model, params, batch['input_ids'],
                                            kwargs={k: v
                                                    for k, v in batch.items()
                                                    if k != "input_ids"}
                                            ).logits
    loss = torch.nn.functional.cross_entropy(logits, batch['labels'])
    weight_decay = 0.0001
    weight_l2 = sum(p.norm()**2 for p in params.values() if p.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    return loss + weight_penalty


@mem.cache
def run_one(fun_name, model_name, batch_size=16, n_reps=1):
    framework = MODEL_DICT[model_name]['framework']
    num_classes = MODEL_DICT[model_name]['num_classes']
    if framework == "jax":
        use_gpu = xla_bridge.get_backend().platform == "gpu"
        key = jax.random.PRNGKey(0)
        if model_name != "bert_flax":
            key, subkey = jax.random.split(key)
            batch = {
                'images': jax.random.normal(key, (batch_size, 224, 224, 3)),
                'labels': jax.random.randint(subkey, (batch_size,), 0,
                                             num_classes)
            }
        else:
            keys = jax.random.split(key, 4)
            batch = {
                'input_ids': jax.random.randint(keys[0], (batch_size, 128),
                                                0, 10000),
                'attention_mask': jax.random.randint(keys[1],
                                                     (batch_size, 128),
                                                     0, 2),
                'token_type_ids': jax.random.randint(keys[2],
                                                     (batch_size, 128),
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
            lambda x: jax.grad(loss_fn_jax)(x, model, batch,
                                            num_classes=num_classes)
        )
    elif framework == "torch":
        use_gpu = torch.cuda.is_available()
        gen = torch.Generator().manual_seed(0)
        if model_name != "bert_torch":
            batch = {
                'images': torch.randn(batch_size, 3, 224, 224, generator=gen),
                'labels': torch.randint(0, num_classes, (batch_size,),
                                        generator=gen)
            }
        else:
            batch = {
                'input_ids': torch.randint(0, 10000, (batch_size, 128),
                                           generator=gen),
                'attention_mask': torch.randint(0, 2, (batch_size, 128),
                                                generator=gen),
                'token_type_ids': torch.randint(0, 2, (batch_size, 128),
                                                generator=gen),
                'position_ids': None,
                'head_mask': None,
                'labels': torch.randint(0, num_classes, (batch_size,),
                                        generator=gen)
            }

        model = MODEL_DICT[model_name]['module'].from_pretrained(
            MODEL_DICT[model_name]['model']
        )
        replace_all_batch_norm_modules_(model)
        if use_gpu:
            batch = {k: v.cuda() for k, v in batch.items()}
            model = model.cuda()
        params = dict(model.named_parameters())

        def grad_fun(x):
            return torch.func.grad(loss_fn_torch)(x, model, batch)

    if fun_name == "grad":
        grad_fun(params)  # First run for compilation
    else:
        v = grad_fun(params)  # First run to get a v and for compilation
        hvp_fun = fun_dict[fun_name]['fun'](model, batch,
                                            num_classes=num_classes,
                                            framework=framework)
        hvp_fun(params, v)  # First run for compilation
    times = []
    memories = []
    for _ in range(n_reps):
        if fun_name == "grad":
            memory = 0
            if framework == "jax":
                start = perf_counter()
                jax.block_until_ready(grad_fun(params))
                time = perf_counter() - start
            elif framework == "torch":
                start = perf_counter()
                grad_fun(params)
                time = perf_counter() - start
            times.append(time)
            memories.append(memory)
        else:
            if framework == "jax":
                start = perf_counter()
                jax.block_until_ready(hvp_fun(params, v))
                time = perf_counter() - start
                start = perf_counter()
                jax.block_until_ready(grad_fun(params))
                grad_time = perf_counter() - start
            elif framework == "torch":
                start = perf_counter()
                hvp_fun(params, v)
                time = perf_counter() - start
                start = perf_counter()
                grad_fun(params)
                grad_time = perf_counter() - start
            times.append(time - grad_time)

    return dict(
        model=model_name.split('_')[0],
        label=fun_name,
        time=times,
        memory=memories,
        batch_size=batch_size,
        framework=framework,
        rep=jnp.arange(n_reps),
    )


def run_bench(fun_list, model_list, n_reps, batch_size_list,
              slurm_config_path=None):
    run = partial(run_one, n_reps=n_reps)

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


def hvp_forward_over_reverse(model, batch, num_classes=1000, framework="jax"):
    """
    Returns the Hessian-vector product operator that uses forward-over-reverse
    propagation.
    """
    if framework == "jax":
        grad_fun = jax.jit(
            lambda x: jax.grad(loss_fn_jax)(x, model, batch,
                                            num_classes=num_classes)
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.jvp(grad_fun, (x, ), (v, ))[1]
        )
    elif framework == "torch":
        def grad_fun(x):
            return torch.func.grad(loss_fn_torch)(x, model, batch)

        def hvp_fun(x, v):
            return torch.func.jvp(grad_fun, (x, ), (v, ))[1]
    return hvp_fun


def hvp_reverse_over_forward(model, batch, num_classes=1000, framework="jax"):
    """
    Returns the Hessian-vector product operator that uses reverse-over-forward
    propagation.
    """
    if framework == 'jax':
        jvp_fun = jax.jit(
            lambda x, v: jax.jvp(
                lambda y: loss_fn_jax(y, model, batch,
                                      num_classes=num_classes),
                (x, ), (v, )
            )[1]
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.grad(jvp_fun)(x, v)
        )
    elif framework == 'torch':
        def jvp_fun(x, v):
            return torch.func.jvp(loss_fn_torch, (x, ), (v, ))[1]

        def hvp_fun(x, v):
            return torch.func.grad(jvp_fun)(x, v)

    return hvp_fun


def hvp_reverse_over_reverse(model, batch, num_classes=1000, framework="jax"):
    """
    Returns the Hessian-vector product operator that uses reverse-over-reverse
    propagation.
    """
    if framework == 'jax':
        grad_fun = jax.jit(
            lambda x: jax.grad(loss_fn_jax)(x, model, batch,
                                            num_classes=num_classes)
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.grad(lambda x: utils.tree_dot(grad_fun(x), v))(x)
        )
    elif framework == 'torch':
        def grad_fun(x):
            return torch.func.grad(loss_fn_torch)(x, model, batch)

        def hvp_fun(x, v):
            return torch.func.grad(lambda x: sum(
                torch.dot(a.ravel(), b.ravel())
                for a, b in zip(grad_fun(x), v))
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
        hvp_forward_over_reverse=dict(fun=hvp_forward_over_reverse,
                                      label="HVP forward-over-reverse"),
        hvp_reverse_over_forward=dict(fun=hvp_reverse_over_forward,
                                      label="HVP reverse-over-forward"),
        hvp_reverse_over_reverse=dict(fun=hvp_reverse_over_reverse,
                                      label="HVP reverse-over-reverse"),
    )
    model_list = [k for k in MODEL_DICT.keys()
                  if MODEL_DICT[k]['framework'] == framework]
    fun_list = fun_dict.keys()
    import ipdb
    ipdb.set_trace()
    # run_one('hvp_forward_over_reverse', 'bert_flax')
    # df = run_bench(fun_list, model_list, n_reps=N_REPS,
    #                batch_size_list=BATCH_SIZE_LIST,
    #                slurm_config_path=SLURM_CONFIG)
    # df.to_parquet(f'../outputs/bench_hvp_neural_network_{framework}.parquet')