import models

import jax
import optax
import jax.numpy as jnp
from flax.training import common_utils

from time import perf_counter


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


def run_one(x, v, hvp, n_reps=10):
    times = jnp.zeros(n_reps)

    for k in range(n_reps):
        start = perf_counter()
        hvp(x, v)
        end = perf_counter() - start
        times = times.at[k].set(end)
    return times.mean(), times.std()


if __name__ == '__main__':
    import pandas as pd

    N_REPS = 20

    MODELS_LIST = [
        models.ResNet18,
        models.ResNet34,
        models.ResNet50,
        models.ResNet101,
        models.ResNet152,
        models.ResNet200,
    ]

    BATCH_SIZE = 16
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    batch = {
        'images': jax.random.normal(key, (BATCH_SIZE, 224, 224, 3)),
        'labels': jax.random.randint(subkey, (BATCH_SIZE,), 0, NUM_CLASSES)
    }

    key, subkey = jax.random.split(key)

    times_means = jnp.zeros((len(MODELS_LIST)))
    times_stds = jnp.zeros((len(MODELS_LIST)))

    grad_times_means = jnp.zeros((len(MODELS_LIST)))
    grad_times_stds = jnp.zeros((len(MODELS_LIST)))

    for i, model_cls in enumerate(MODELS_LIST):
        print(i)
        model = model_cls(num_classes=NUM_CLASSES)
        init = model.init(key, batch['images'], train=True)
        grad_fn = jax.jit(
            lambda x: jax.grad(loss_fn)(x, model, batch, init['batch_stats'])
        )
        hvp_fun = jax.jit(
            lambda x, v: jax.jvp(grad_fn, (x, ), (v, ))[1]
        )
        v = grad_fn(init['params'])
        hvp_fun(init['params'], v)
        print('run hvp')
        mean, std = run_one(init['params'], v, hvp_fun, n_reps=N_REPS)
        times_means = times_means.at[i].set(mean)
        times_stds = times_stds.at[i].set(std)

        print('run grad')
        mean, std = run_one(init['params'], v,
                            jax.jit(lambda x, v: grad_fn(x)),
                            n_reps=N_REPS)
        grad_times_means = grad_times_means.at[i].set(mean)
        grad_times_stds = grad_times_stds.at[i].set(std)

    pd.DataFrame(
        {'mean': times_means, 'std': times_stds},
    ).to_parquet('hvp_df.parquet')
    pd.DataFrame(
        {'mean': grad_times_means, 'std': grad_times_stds},
    ).to_parquet('grad_df.parquet')
