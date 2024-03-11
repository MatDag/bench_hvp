import os
import pandas as pd

from memory_monitor import GPUMemoryMonitor

from transformers.utils import logging

logging.set_verbosity_error()

try:
    import utils_jax
    from utils_jax import JAX_MODELS
except ImportError:
    JAX_MODELS = {}

# Torch framework
try:
    import utils_torch
    from utils_torch import TORCH_MODELS
except ImportError:
    TORCH_MODELS = {}


MODEL_DICT = dict(**JAX_MODELS, **TORCH_MODELS)


def run_one(fun_name, model_name, batch_size=16):
    framework = MODEL_DICT[model_name]['framework']
    num_classes = MODEL_DICT[model_name]['num_classes']

    utils = utils_jax if framework == "jax" else utils_torch
    model, params, batch = utils.get_model_and_batch(
        model_name, batch_size, num_classes, key=0
    )
    grad_fun = utils.get_grad(model, batch)

    if fun_name != "grad":
        v = params.copy()
        hvp_fun = getattr(utils, f"get_{fun_name}")(model, batch)
    monitor = GPUMemoryMonitor()
    if fun_name == "grad":
        grad_fun(params)
    else:
        hvp_fun(params, v)
    monitor.join()
    memory = max(monitor.memory_buffer)

    return memory


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', '-f', type=str, default='jax',
                        choices=['jax', 'torch'])
    parser.add_argument('--jit', '-j', type=str, default="activate")
    parser.add_argument('--fun', type=str, default='grad',
                        choices=['grad', 'hvp_forward_over_reverse',
                                 'hvp_reverse_over_forward',
                                 'hvp_reverse_over_reverse'])
    parser.add_argument('--model', '-m', type=str, default='resnet50_flax')
    parser.add_argument('--batch_size', '-b', type=int, default=16)

    framework = parser.parse_args().framework
    jit_fun = parser.parse_args().jit == "activate"
    fun_name = parser.parse_args().fun
    model_name = parser.parse_args().model
    batch_size = parser.parse_args().batch_size
    model_list = [k for k in MODEL_DICT.keys()
                  if MODEL_DICT[k]['framework'] == framework]

    jax_and_no_jit = framework == "jax" and not jit_fun
    if jax_and_no_jit:
        from jax.config import config
        config.update('jax_disable_jit', True)
        path_out = '../outputs/bench_hvp_memory_jax_without_jit.parquet'
    else:
        path_out = f'../outputs/bench_hvp_memory_{framework}.parquet'

    print(f'Running {fun_name} on {model_name} ' +
          f'with batch size {batch_size}.')

    memory = run_one(fun_name, model_name, batch_size)
    memory /= 1024**2
    print(f"Peak memory usage: {memory:.2f} MiB")
    if os.path.exists(path_out):
        df = pd.read_parquet(path_out)
        df.loc[
            (model_name.split('_')[0], fun_name, batch_size, framework),
            'memory'
        ] = memory
    else:
        df = pd.DataFrame(dict(
            model=[model_name.split('_')[0]],
            fun=[fun_name],
            batch_size=[batch_size],
            memory=[memory],
            framework=[framework]
        )).set_index(['model', 'fun', 'batch_size', 'framework'])

    df.to_parquet(path_out)
