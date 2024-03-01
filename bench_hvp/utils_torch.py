
import torch

from transformers import ViTForImageClassification
from transformers import ResNetForImageClassification
from transformers import BertForSequenceClassification

BATCH_NORM_NOOP = True


TORCH_MODELS = dict(
    resnet34_torch=dict(
        module=ResNetForImageClassification, model="microsoft/resnet-34",
        framework="torch", num_classes=1000
    ),
    resnet50_torch=dict(
        module=ResNetForImageClassification, model="microsoft/resnet-50",
        framework="torch", num_classes=1000
    ),
    vit_torch=dict(
        module=ViTForImageClassification, model="google/vit-base-patch16-224",
        framework="torch", num_classes=1000
    ),
    bert_torch=dict(
        module=BertForSequenceClassification, model="bert-base-uncased",
        framework="torch", num_classes=2
    ),
)


def replace_all_batch_norm_by_noop(root):
    """There are some memory issue with batch norm, so we remove them."""
    # base case
    if isinstance(root, torch.nn.modules.batchnorm._BatchNorm):
        root.running_mean = None
        root.running_var = None
        root.num_batches_tracked = None
        root.track_running_stats = False
        if BATCH_NORM_NOOP:
            root.forward = lambda x: x

    # Recursively replace all batch norm modules
    for obj in root.children():
        replace_all_batch_norm_by_noop(obj)
    return root


def loss_fn(params, model, batch):
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


def get_model_and_batch(model_name, batch_size, num_classes=1000,
                        sequence_length=32, key=0):
    gen = torch.Generator().manual_seed(key)

    if model_name != "bert_torch":
        image_size = 96 if model_name == "vit_torch" else 224
        batch = {
            'images': torch.randn(batch_size, 3, image_size,
                                  image_size, generator=gen),
            'labels': torch.randint(
                0, num_classes, (batch_size,), generator=gen
            )
        }
    else:
        batch = {
            'input_ids': torch.randint(
                0, 10000, (batch_size, sequence_length), generator=gen
            ),
            'attention_mask': torch.randint(
                0, 2, (batch_size, sequence_length), generator=gen
            ),
            'token_type_ids': torch.randint(
                0, 2, (batch_size, sequence_length), generator=gen
            ),
            'position_ids': None,
            'head_mask': None,
            'labels': torch.randint(
                0, num_classes, (batch_size,), generator=gen
            )
        }

    model = TORCH_MODELS[model_name]['module'].from_pretrained(
        TORCH_MODELS[model_name]['model']
    )
    replace_all_batch_norm_by_noop(model)

    if model_name == "vit_torch":
        config = model.config
        config.image_size = image_size
        model = TORCH_MODELS[model_name]['module'](config)

    if torch.cuda.is_available():
        model = model.cuda()
        batch = {k: v.cuda() for k, v in batch.items() if v is not None}

    params = dict(model.named_parameters())

    return model, params, batch


def cuda_synchronize(f):
    """Decorator to call cuda.synchronize before return."""
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        torch.cuda.synchronize()
        return res
    return wrapper


def get_grad(model, batch):
    """
    Returns the Hessian-vector product operator that uses reverse-over-reverse
    propagation.
    """
    def f(x):
        return loss_fn(x, model, batch)

    return cuda_synchronize(torch.func.grad(f))


def get_hvp_forward_over_reverse(model, batch):
    """
    Returns the Hessian-vector product operator that uses forward-over-reverse
    propagation.
    """
    def f(x):
        return loss_fn(x, model, batch)

    grad_fun = torch.func.grad(f)

    def hvp_fun(x, v):
        return torch.func.jvp(grad_fun, (x,), (v,))[1]
    return cuda_synchronize(hvp_fun)


def get_hvp_reverse_over_forward(model, batch):
    """
    Returns the Hessian-vector product operator that uses reverse-over-forward
    propagation.
    """
    def f(x):
        return loss_fn(x, model, batch)

    def jvp_fun(x, v):
        return torch.func.jvp(f, (x,), (v,))[1]

    hvp_fun = torch.func.grad(jvp_fun)

    return cuda_synchronize(hvp_fun)


def get_hvp_reverse_over_reverse(model, batch):
    """
    Returns the Hessian-vector product operator that uses reverse-over-reverse
    propagation.
    """
    def f(x):
        return loss_fn(x, model, batch)

    grad_fun = torch.func.grad(f)

    hvp_fun = torch.func.grad(
        lambda x, v: sum(
            torch.dot(a.ravel(), b.ravel())
            for a, b in zip(grad_fun(x).values(), v.values())
        ),
        argnums=0
    )

    return cuda_synchronize(hvp_fun)
