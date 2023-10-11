import jax
import jax.numpy as jnp


def tree_dot(a, b):
    """
    Compute the inner product between two PyTrees.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(jnp.sum, jax.tree_map(jnp.multiply, a, b)),
        0.
    )
