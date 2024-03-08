# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

dataset = 'gp'
prompt_dir = "/home/mpacek/data/prompts/"
wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-gp-8192'
out_dir = f'out-{wandb_run_name}'

# these make the total batch size be ~0.5M
# 12 batch size * 8192 block size * 5 gradaccum * 8 GPUs ~= 4M
batch_size = 1
block_size = 8192
gradient_accumulation_steps = 5 * 8 * 12

# this makes total number of tokens be 2400B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
