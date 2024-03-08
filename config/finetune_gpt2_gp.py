eval_interval = 1000
eval_iters = 40
log_interval = 10
wandb_log = True # feel free to turn on
wandb_project = 'owt'
wandb_run_name = 'ft-gpt2-gp-1024'
out_dir = f'out-{wandb_run_name}'
prompt_dir = "/home/mpacek/data/prompts/"

dataset = 'gp'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 8 batch_size * 8 grad_accum * 1024 tokens = 65,536 tokens/iter
# gp has 2,334,931,925 tokens, so 1 epoch ~= 35628 iters
batch_size = 8
gradient_accumulation_steps = 8
max_iters = 40000

# finetune at constant LR
learning_rate = 2 * 3e-5
decay_lr = False
dropout = 0.1
