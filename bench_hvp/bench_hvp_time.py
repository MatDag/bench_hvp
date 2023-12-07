import jax
import optax
import jax.numpy as jnp
from jax.lib import xla_bridge
from flax.training import common_utils

import torch
from functorch.experimental import replace_all_batch_norm_modules_

import numpy as np
import pandas as pd
from functools import partial

import itertools
from time import perf_counter

from transformers import ViTForImageClassification
from transformers import FlaxViTForImageClassification

from transformers import ResNetForImageClassification
from transformers import FlaxResNetForImageClassification

from transformers import BertForSequenceClassification
from transformers import FlaxBertForSequenceClassification

import utils

from joblib import Memory
mem = Memory(location='__cache__')

MODEL_DICT = dict(
    resnet34_flax=dict(module=FlaxResNetForImageClassification,
                       model="microsoft/resnet-34", framework="jax",
                       num_classes=1000),
    resnet50_flax=dict(module=FlaxResNetForImageClassification,
                       model="microsoft/resnet-50", framework="jax",
                       num_classes=1000),
    vit_flax=dict(module=FlaxViTForImageClassification,
                  model="google/vit-base-patch16-224", framework="jax",
                  num_classes=1000),
    bert_flax=dict(module=FlaxBertForSequenceClassification,
                   model="bert-base-uncased", framework="jax", num_classes=2),
    resnet34_torch=dict(module=ResNetForImageClassification,
                        model="microsoft/resnet-34", framework="torch",
                        num_classes=1000),
    resnet50_torch=dict(module=ResNetForImageClassification,
                        model="microsoft/resnet-50", framework="torch",
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
def run_one(model_name, batch_size, fun_name, n_reps=1):
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
            batch = {k: v.cuda() for k, v in batch.items() if v is not None}
            model = model.cuda()
        params = dict(model.named_parameters())

        def grad_fun(x):
            def f(y):
                return loss_fn_torch(y, model, batch)
            return torch.func.grad(f)(x)

    # We compute the quantities a first time for computation
    if fun_name == "grad":
        if framework == "jax":
            jax.block_until_ready(grad_fun(params))
        else:
            grad_fun(params)
    else:
        v = grad_fun(params)
        hvp_fun = fun_dict[fun_name](model, batch,
                                     num_classes=num_classes,
                                     framework=framework)
        if framework == "jax":
            jax.block_until_ready(hvp_fun(params, v))
        else:
            hvp_fun(params, v)

    times = []
    if framework == "torch":
        torch.cuda.synchronize()
    for i in range(n_reps):
        print(
            f"Running {fun_name} on {model_name}:{batch_size} "
            f"-- {i/n_reps:.2%}\r", end='', flush=True
        )
        if fun_name == "grad":
            start = perf_counter()
            if framework == "jax":
                jax.block_until_ready(grad_fun(params))
            elif framework == "torch":
                grad_fun(params)
                torch.cuda.synchronize()
            time = perf_counter() - start
            times.append(time)
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
                torch.cuda.synchronize()
                time = perf_counter() - start
                start = perf_counter()
                grad_fun(params)
                torch.cuda.synchronize()
                grad_time = perf_counter() - start

            times.append(time - grad_time)

    print(f"Running {fun_name} on {model_name}:{batch_size} -- done   ")
    return dict(
        model=model_name.split('_')[0],
        fun=fun_name,
        time=times,
        batch_size=batch_size,
        framework=framework,
        rep=np.arange(n_reps),
    )


def run_bench(fun_list, model_list, n_reps, batch_size_list,
              slurm_config_path=None):
    run = partial(run_one, n_reps=n_reps)
    all_runs = list(itertools.product(model_list, batch_size_list, fun_list))
    skip = []

    jobs = [run(*args) for args in all_runs if args[:-1] not in skip]

    return pd.concat([pd.DataFrame(res) for res in jobs])


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
            lambda x, v: jax.jvp(grad_fun, (x,), (v,))[1]
        )
    elif framework == "torch":
        def f(x):
            return loss_fn_torch(x, model, batch)

        grad_fun = torch.func.grad(f)

        def hvp_fun(x, v):
            return torch.func.jvp(grad_fun, (x,), (v,))[1]
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
                (x,), (v,)
            )[1]
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.grad(jvp_fun)(x, v)
        )
    elif framework == 'torch':
        def f(x):
            return loss_fn_torch(x, model, batch)

        def jvp_fun(x, v):
            return torch.func.jvp(f, (x,), (v,))[1]

        hvp_fun = torch.func.grad(jvp_fun)

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
            lambda x, v: jax.grad(lambda y: utils.tree_dot(grad_fun(y), v))(x)
        )
    elif framework == 'torch':
        def f(x):
            return loss_fn_torch(x, model, batch)

        grad_fun = torch.func.grad(f)

        hvp_fun = torch.func.grad(
            lambda x, v: sum(
                torch.dot(a.ravel(), b.ravel())
                for a, b in zip(grad_fun(x).values(), v.values())
            ),
            argnums=0
        )

    return hvp_fun


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', '-f', type=str, default='jax',
                        choices=['jax', 'torch'])
    parser.add_argument('--n_reps', '-n', type=int, default=100)
    parser.add_argument('--config', '-c', type=str, default=SLURM_CONFIG)

    args = parser.parse_args()
    framework = args.framework
    n_reps = args.n_reps
    slurm_config = args.config

    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]

    fun_dict = dict(
        grad=dict(fun=None, label="Gradient"),
        hvp_forward_over_reverse=hvp_forward_over_reverse,
        hvp_reverse_over_forward=hvp_reverse_over_forward,
        hvp_reverse_over_reverse=hvp_reverse_over_reverse,
    )
    model_list = [k for k in MODEL_DICT.keys()
                  if MODEL_DICT[k]['framework'] == framework]
    model_list = ['resnet50_torch']
    fun_list = fun_dict.keys()

    df = run_bench(fun_list, model_list, n_reps=n_reps,
                   batch_size_list=batch_size_list,
                   slurm_config_path=slurm_config)
    df.to_parquet(f'../outputs/bench_hvp_time_{framework}.parquet')
