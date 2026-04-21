# Baseline config for nanoGPT on TinyStories (GPT-2 BPE via tiktoken)
# Intended to be launched via template.sbatch which overrides:
#   --dataset, --seed, --out_dir, --device=cuda, --compile=False, --wandb_log=False

# -----------------------------------------------------------------------------
# I/O / logging
log_interval = 10
eval_interval = 500
eval_iters = 200
eval_only = False
save_best_to_different = True

# -----------------------------------------------------------------------------
# Initialization
init_from = "scratch"

# -----------------------------------------------------------------------------
# Data
# template.sbatch will pass --dataset=tinystories, but keep it here for clarity
dataset = "tinystories"
block_size = 512

# Effective batch = gradient_accumulation_steps * batch_size
# tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
# 4 * 32 * 512 = 65536 tokens / update
batch_size = 32
gradient_accumulation_steps = 4


# -----------------------------------------------------------------------------
# Model
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# BPE vocab (GPT-2 is 50257; 50304 = rounded up for efficiency, как в openwebtext-конфигах)
vocab_size = 50304

# -----------------------------------------------------------------------------
# optimizer
learning_rate = 1e-3
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# -----------------------------------------------------------------------------
# schedule
decay_lr = True
max_iters = 20000
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 5e-5
