from models import ResNet, BottleneckResNetBlock

import jax
import optax
import jax.numpy as jnp
from flax.training import common_utils

import pandas as pd
from functools import partial

from time import perf_counter

import tqdm
from joblib import Memory, Parallel, delayed
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
def run_one(size, n_reps=10, batch_size=16, num_classes=1000):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    batch = {
        'images': jax.random.normal(key, (batch_size, 224, 224, 3)),
        'labels': jax.random.randint(subkey, (batch_size,), 0, num_classes)
    }

    key, subkey = jax.random.split(key)
    model = init_model(size)
    init = model.init(key, batch['images'], train=True)
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch, init['batch_stats'])
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.jvp(grad_fun, (x, ), (v, ))[1]
    )
    v = grad_fun(init['params'])  # First run to get a v and for compilation
    hvp_fun(init['params'], v)  # First run for compilation

    hvp_times = jnp.zeros(n_reps)
    grad_times = jnp.zeros(n_reps)

    for k in range(n_reps):
        start = perf_counter()
        hvp_fun(init['params'], v)
        end = perf_counter() - start
        hvp_times = hvp_times.at[k].set(end)
        start = perf_counter()
        grad_fun(init['params'])
        end = perf_counter() - start
        grad_times = grad_times.at[k].set(end)

    hvp_mean = hvp_times.mean()
    hvp_std = hvp_times.std()
    grad_mean = grad_times.mean()
    grad_std = grad_times.std()
    return dict(
        deepth=2+3*(3+8+size+3), hvp_mean_time=hvp_mean, hvp_std_time=hvp_std,
        grad_mean_time=grad_mean, grad_std_time=grad_std,
    )


def init_model(size):
    model = partial(ResNet, stage_sizes=[3, 8, size, 3],
                    block_cls=BottleneckResNetBlock)
    return model


def run_bench(sizes, batch_size=16, n_reps=10, num_classes=1000, n_jobs=1):
    run = partial(run_one, n_reps=n_reps, batch_size=batch_size,
                  num_classes=num_classes)

    res = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run)(size) for size in tqdm.tqdm(sizes)
    )
    all_results = []
    for r in res:
        all_results.extend(r)

    return pd.DataFrame(all_results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Benchmark for logreg hyperparameter selection.'
    )

    parser.add_argument('--n-jobs', '-j', type=int, default=1,
                        help='# of parallel runs.')

    parser.add_argument('--n-reps', '-r', type=int, default=1,
                        help='# of repetitions.')

    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='Batch size.')

    args = parser.parse_args()

    n_reps = args.n_reps
    batch_size = args.batch_size
    n_jobs = args.n_jobs

    SIZES = jnp.arange(2, 50, 25)

    df = run_bench(SIZES, batch_size=batch_size, n_reps=n_reps, n_jobs=n_jobs)
    df.to_parquet('../outputs/bench_hvp.parquet')
