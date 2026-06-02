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
model_path = "/home/alaltischenko/proj/tinyllm/fine-tune-2/runs/3654739_3_ft2_F_base_2026-02-26_06-48-53/out/ckpt_best.pt"

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

# =============================================================================
# F_15: LoRA_attention_mlp_h_frozen_tokens_frozen
# Pair: F_7 = fake-qLoRA_attention_mlp_h_frozen_tokens_frozen from the old run
# Purpose: LoRA counterpart of the already completed F_7.
# Trainable now: LoRA A/B for attention+MLP + ln_f.
# Frozen now:    wte/wpe + original h[i], including ln_1/ln_2.
# Caveat:        F_7's actual trainable-parameter count suggests no ln_f was trainable
#                in that old run. Current train.py cannot reproduce that exactly by
#                config only; add freeze_ln_f=True for a strict pair.
# =============================================================================
qlora_enable = False

lora_enable = True
lora_targets = "all-linear"
lora_target_layers = "all"
lora_rank = 8
lora_alpha = 1.0
lora_bias = False
lora_merge_weights = True

# Freeze h[i] before inserting LoRA, but apply no quantization.
quant_enable = True
quant_targets = ""
quant_target_layers = ""
quant_freeze_base = True

freeze_n_layers = 0
freeze_embeddings = True # weight tying is applying

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
