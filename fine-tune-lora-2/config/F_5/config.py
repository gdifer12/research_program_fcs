# -----------------------------------------------------------------------------
# I/O / logging
log_interval = 10
eval_interval = 500
eval_iters = 200
eval_only = False
save_best_to_different = True

# -----------------------------------------------------------------------------
# Initialization
init_from = 'resume'
model_path = "fine-tune-2/runs/3654739_3_ft2_F_base_2026-02-26_06-48-53/out/ckpt_best.pt"

# -----------------------------------------------------------------------------
# Data
dataset = "tinystoriesInstruct"
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

# LoRa settings
lora_enable = True
lora_targets = "wte"
lora_target_layers = ""
lora_rank = 8
lora_alpha = 8.0
lora_bias = False

# -----------------------------------------------------------------------------
# optimizer
learning_rate = 3e-4
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5

# -----------------------------------------------------------------------------
# schedule
decay_lr = True
max_iters = 20000 + 10000
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 3e-5
