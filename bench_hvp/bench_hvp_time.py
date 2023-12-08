import itertools
import numpy as np
import pandas as pd
from joblib import Memory
from time import perf_counter


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


mem = Memory(location='__cache__')


@mem.cache
def run_one(model_name, batch_size, fun_name, n_reps=1):
    framework = MODEL_DICT[model_name]['framework']
    num_classes = MODEL_DICT[model_name]['num_classes']

    utils = utils_jax if framework == "jax" else utils_torch
    model, params, batch = utils.get_model_and_batch(
        model_name, batch_size, num_classes, key=0
    )
    grad_fun = utils.get_grad(model, batch)

    # We compute the quantities a first time for computation
    if fun_name == "grad":
        grad_fun(params)

        def get_time():
            start = perf_counter()
            grad_fun(params)
            time = perf_counter() - start
            return time
    else:
        v = grad_fun(params)
        hvp_fun = getattr(utils, f"get_{fun_name}")(model, batch)
        hvp_fun(params, v)

        def get_time():
            start = perf_counter()
            hvp_fun(params, v)
            time = perf_counter() - start
            start = perf_counter()
            grad_fun(params)
            grad_time = perf_counter() - start
            return time - grad_time

    times = []
    for i in range(n_reps):
        print(
            f"Running {fun_name} on {model_name}:{batch_size} "
            f"-- {i/n_reps:.2%}\r", end='', flush=True
        )
        times.append(get_time())

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
    all_runs = list(itertools.product(model_list, batch_size_list, fun_list))
    skip = [("resnet50_torch", 160, "hvp_reverse_over_reverse")]

    jobs = [
        run_one(*args, n_reps=n_reps) for args in all_runs if args not in skip
    ]

    return pd.concat([pd.DataFrame(res) for res in jobs])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', '-f', type=str, default='jax',
                        choices=['jax', 'torch'])
    parser.add_argument('--n_reps', '-n', type=int, default=100)
    parser.add_argument('--config', '-c', type=str, default='config/slurm.yml')

    args = parser.parse_args()
    framework = args.framework

    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]

    fun_list = [
        "grad", "hvp_forward_over_reverse", "hvp_reverse_over_forward",
        "hvp_reverse_over_reverse"
    ]
    model_list = [k for k in MODEL_DICT.keys()
                  if MODEL_DICT[k]['framework'] == framework]
    model_list = ['resnet50_torch']

    df = run_bench(fun_list, model_list, n_reps=args.n_reps,
                   batch_size_list=batch_size_list[::-1],
                   slurm_config_path=args.config)
    df.to_parquet(f'../outputs/bench_hvp_time_{framework}.parquet')
