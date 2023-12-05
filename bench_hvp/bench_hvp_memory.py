import jax
import optax
import jax.numpy as jnp
from jax.lib import xla_bridge
from flax.training import common_utils

import torch
from functorch.experimental import replace_all_batch_norm_modules_

from transformers import ViTForImageClassification
from transformers import FlaxViTForImageClassification

from transformers import ResNetForImageClassification
from transformers import FlaxResNetForImageClassification

from transformers import BertForSequenceClassification
from transformers import FlaxBertForSequenceClassification

from memory_monitor import GPUMemoryMonitor

import utils

import os

import pandas as pd

MODEL_DICT = dict(
    resnet50_flax=dict(module=FlaxResNetForImageClassification,
                       model="microsoft/resnet-50", framework="jax",
                       num_classes=1000),
    resnet34_flax=dict(module=FlaxResNetForImageClassification,
                       model="microsoft/resnet-34", framework="jax",
                       num_classes=1000),
    vit_flax=dict(module=FlaxViTForImageClassification,
                  model="google/vit-base-patch16-224", framework="jax",
                  num_classes=1000),
    bert_flax=dict(module=FlaxBertForSequenceClassification,
                   model="bert-base-uncased", framework="jax", num_classes=2),
    resnet50_torch=dict(module=ResNetForImageClassification,
                        model="microsoft/resnet-50", framework="torch",
                        num_classes=1000),
    resnet34_torch=dict(module=ResNetForImageClassification,
                        model="microsoft/resnet-34", framework="torch",
                        num_classes=1000),
    vit_torch=dict(module=ViTForImageClassification,
                   model="google/vit-base-patch16-224", framework="torch",
                   num_classes=1000),
    bert_torch=dict(module=BertForSequenceClassification,
                    model="bert-base-uncased", framework="torch",
                    num_classes=2),
)


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


def run_one(fun_name, model_name, batch_size=16):
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

    if fun_name != "grad":
        v = params.copy()
    monitor = GPUMemoryMonitor()
    fun = fun_dict[fun_name]['fun'](model, batch,
                                    num_classes=num_classes,
                                    framework=framework)
    if fun_name == "grad":
        if framework == "jax":
            jax.block_until_ready(fun(params))
        else:
            fun(params)
    else:
        if framework == "jax":
            jax.block_until_ready(fun(params, v))
        else:
            fun(params, v)
    monitor.join()
    memory = max(monitor.memory_buffer)

    return memory


def grad_fun(model, batch, num_classes=1000, framework="jax"):
    """
    Returns the gradient operator.
    """
    if framework == "jax":
        grad_fun = jax.jit(
            lambda x: jax.grad(loss_fn_jax)(x, model, batch,
                                            num_classes=num_classes)
        )
    elif framework == "torch":
        def grad_fun(x):
            return torch.func.grad(loss_fn_torch)(x, model, batch)
    return grad_fun


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
            )[1])
        hvp_fun = jax.jit(
            lambda x, v: jax.grad(jvp_fun)(x, v)
        )
    elif framework == 'torch':
        def jvp_fun(x, v):
            return torch.func.jvp(lambda y: loss_fn_torch(y, model, batch),
                                  (x, ), (v, ))[1]

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
                for a, b in zip(grad_fun(x).values(), v.values()))
            )(x)

    return hvp_fun


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', '-f', type=str, default='jax',
                        choices=['jax', 'torch'])
    parser.add_argument('--fun', type=str, default='grad',
                        choices=['grad', 'hvp_forward_over_reverse',
                                 'hvp_reverse_over_forward',
                                 'hvp_reverse_over_reverse'])
    parser.add_argument('--model', '-m', type=str, default='resnet50_flax')
    parser.add_argument('--batch_size', '-b', type=int, default=16)

    framework = parser.parse_args().framework
    fun_name = parser.parse_args().fun
    model_name = parser.parse_args().model
    batch_size = parser.parse_args().batch_size

    fun_dict = dict(
        grad=dict(fun=grad_fun, label="Gradient"),
        hvp_forward_over_reverse=dict(fun=hvp_forward_over_reverse,
                                      label="HVP forward-over-reverse"),
        hvp_reverse_over_forward=dict(fun=hvp_reverse_over_forward,
                                      label="HVP reverse-over-forward"),
        hvp_reverse_over_reverse=dict(fun=hvp_reverse_over_reverse,
                                      label="HVP reverse-over-reverse"),
    )
    model_list = [k for k in MODEL_DICT.keys()
                  if MODEL_DICT[k]['framework'] == framework]

    print(f'Running {fun_dict[fun_name]["label"]} on {model_name} ' +
          f'with batch size {batch_size}.')

    memory = run_one(fun_name, model_name, batch_size)
    memory /= 1024**2
    print(f"Peak memory usage: {memory:.2f} MiB")
    if os.path.exists(f'../outputs/bench_hvp_memory_{framework}.parquet'):
        df = pd.read_parquet(
            f'../outputs/bench_hvp_memory_{framework}.parquet'
        )
        df.loc[
            (model_name.split('_')[0], fun_name, batch_size), 'memory'
        ] = memory
    else:
        df = pd.DataFrame(dict(
            model=[model_name.split('_')[0]],
            fun=[fun_name],
            batch_size=[batch_size],
            memory=[memory]
        )).set_index(['model', 'fun', 'batch_size'])

    df.to_parquet(f'../outputs/bench_hvp_memory_{framework}.parquet')
