import jax
import optax
import jax.numpy as jnp
from flax.training import common_utils

from transformers import FlaxViTForImageClassification
from transformers import FlaxResNetForImageClassification
from transformers import FlaxBertForSequenceClassification

JAX_MODELS = dict(
    resnet34_flax=dict(
        module=FlaxResNetForImageClassification, model="microsoft/resnet-34",
        framework="jax", num_classes=1000
    ),
    resnet50_flax=dict(
        module=FlaxResNetForImageClassification, model="microsoft/resnet-50",
        framework="jax", num_classes=1000
    ),
    vit_flax=dict(
        module=FlaxViTForImageClassification,
        model="google/vit-base-patch16-224", framework="jax", num_classes=1000
    ),
    bert_flax=dict(
        module=FlaxBertForSequenceClassification, model="bert-base-uncased",
        framework="jax", num_classes=2
    )
)


def tree_dot(a, b):
    """
    Compute the inner product between two PyTrees.
    """
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(jnp.sum, jax.tree_map(jnp.multiply, a, b)),
        0.
    )


def cross_entropy_loss(logits, labels):
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(xentropy)


def loss_fn(params, model, batch):
    """loss function used for training."""
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    if 'images' in inputs.keys():
        logits = model._module.apply(params, inputs['images']).logits
    else:
        logits = model._module.apply(params, **inputs).logits
    loss = cross_entropy_loss(logits, batch['labels'])
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x ** 2)
                    for x in weight_penalty_params
                    if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss


def get_batch(model_name, batch_size, num_classes=1000, key=0):

    key = jax.random.PRNGKey(key)
    if model_name != "bert_flax":
        key, subkey = jax.random.split(key)
        batch = {
            'images': jax.random.normal(key, (batch_size, 224, 224, 3)),
            'labels': common_utils.onehot(jax.random.randint(
                subkey, (batch_size,), 0, num_classes
            ), num_classes=num_classes)
        }
    else:
        keys = jax.random.split(key, 4)
        batch = {
            'input_ids': jax.random.randint(
                keys[0], (batch_size, 128), 0, 10000
            ),
            'attention_mask': jax.random.randint(
                keys[1], (batch_size, 128), 0, 2
            ),
            'token_type_ids': jax.random.randint(
                keys[2], (batch_size, 128), 0, 2
            ),
            'position_ids': None,
            'head_mask': None,
            'labels': common_utils.onehot(jax.random.randint(
                keys[3], (batch_size,), 0, num_classes
            ), num_classes=num_classes)
        }

    model = JAX_MODELS[model_name]['module'].from_pretrained(
        JAX_MODELS[model_name]['model']
    )
    params = model.params
    if "params" not in params.keys():
        params = {"params": params}
    return model, params, batch


def get_grad(model, batch):
    """
    Returns the Hessian-vector product operator that uses reverse-over-reverse
    propagation.
    """
    grad = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch)
    )
    return lambda x: jax.block_until_ready(grad(x))


def get_hvp_forward_over_reverse(model, batch):
    """
    Returns the Hessian-vector product operator that uses forward-over-reverse
    propagation.
    """
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.jvp(grad_fun, (x,), (v,))[1]
    )
    return lambda x, v: jax.block_until_ready(hvp_fun(x, v))


def get_hvp_reverse_over_forward(model, batch):
    """
    Returns the Hessian-vector product operator that uses reverse-over-forward
    propagation.
    """
    jvp_fun = jax.jit(
        lambda x, v: jax.jvp(
            lambda y: loss_fn(y, model, batch), (x,), (v,)
        )[1]
    )
    hvp_fun = jax.jit(lambda x, v: jax.grad(jvp_fun)(x, v))
    return lambda x, v: jax.block_until_ready(hvp_fun(x, v))


def get_hvp_reverse_over_reverse(model, batch):
    """
    Returns the Hessian-vector product operator that uses reverse-over-reverse
    propagation.
    """
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn)(x, model, batch)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.grad(lambda y: tree_dot(grad_fun(y), v))(x)
    )
    return lambda x, v: jax.block_until_ready(hvp_fun(x, v))
