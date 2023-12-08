#!/bin/bash
batch_size_list=(128  64  32  16  8 4)

source /scratch/mdagreou/conda/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate bench_hvp

python bench_hvp_memory.py -b 128 -m resnet50_torch -f torch --fun grad
python bench_hvp_memory.py -b 128 -m resnet50_torch -f torch --fun hvp_forward_over_reverse
python bench_hvp_memory.py -b 128 -m resnet50_torch -f torch --fun hvp_reverse_over_forward
python bench_hvp_memory.py -b 128 -m resnet50_torch -f torch --fun hvp_reverse_over_reverse

for batch_size in "${batch_size_list[@]}";
do
    python bench_hvp_memory.py -b batch_size -m resnet34_torch -f torch --fun grad
    python bench_hvp_memory.py -b batch_size -m resnet34_torch -f torch --fun hvp_forward_over_reverse
    python bench_hvp_memory.py -b batch_size -m resnet34_torch -f torch --fun hvp_reverse_over_forward
    python bench_hvp_memory.py -b batch_size -m resnet34_torch -f torch --fun hvp_reverse_over_reverse
done

python bench_hvp_memory.py -b 16 -m bert_torch -f torch --fun grad
python bench_hvp_memory.py -b 16 -m bert_torch -f torch --fun hvp_forward_over_reverse
python bench_hvp_memory.py -b 16 -m bert_torch -f torch --fun hvp_reverse_over_forward
python bench_hvp_memory.py -b 16 -m bert_torch -f torch --fun hvp_reverse_over_reverse

python bench_hvp_memory.py -b 16 -m vit_torch -f torch --fun grad
python bench_hvp_memory.py -b 16 -m vit_torch -f torch --fun hvp_forward_over_reverse
python bench_hvp_memory.py -b 16 -m vit_torch -f torch --fun hvp_reverse_over_forward
python bench_hvp_memory.py -b 16 -m vit_torch -f torch --fun hvp_reverse_over_reverse

conda deactivate
conda activate bench_hvp_jax

XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m resnet50_flax -f jax --fun grad
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m resnet50_flax -f jax --fun hvp_forward_over_reverse
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m resnet50_flax -f jax --fun hvp_reverse_over_forward
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m resnet50_flax -f jax --fun hvp_reverse_over_reverse

for batch_size in "${batch_size_list[@]}";
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b batch_size -m resnet34_flax -f jax --fun grad
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b batch_size -m resnet34_flax -f jax --fun hvp_forward_over_reverse
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b batch_size -m resnet34_flax -f jax --fun hvp_reverse_over_forward
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b batch_size -m resnet34_flax -f jax --fun hvp_reverse_over_reverse
done

XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m bert_flax -f jax --fun grad
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m bert_flax -f jax --fun hvp_forward_over_reverse
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m bert_flax -f jax --fun hvp_reverse_over_forward
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 128 -m bert_flax -f jax --fun hvp_reverse_over_reverse

XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 64 -m vit_flax -f jax --fun grad
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 64 -m vit_flax -f jax --fun hvp_forward_over_reverse
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 64 -m vit_flax -f jax --fun hvp_reverse_over_forward
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform  python bench_hvp_memory.py -b 64 -m vit_flax -f jax --fun hvp_reverse_over_reverse

conda deactivate
