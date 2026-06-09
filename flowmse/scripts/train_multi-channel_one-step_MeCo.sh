#!/usr/bin/env bash
set -euo pipefail

NPROC=2                                           
BATCH_PER_GPU=2                                   
LOG_DIR="lightning_logs"

# export NCCL_DEBUG=INFO 

# # Disables direct GPU-to-GPU communication via PCIe (P2P), forcing data through CPU RAM.
# export NCCL_P2P_DISABLE=1
# # Disables InfiniBand (often causing hangs on single-node consumer setups).
# export NCCL_IB_DISABLE=1

# ==== Suggested Environment Variables (NCCL/Threads) ====
export CUDA_VISIBLE_DEVICES=0,1
# export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

# ==== Start Training ====
torchrun --standalone --nproc_per_node="${NPROC}" \
  train.py \
  --backbone ncspp_mixture_concat_multichannel \
  --ode flowmatching \
  --batch_size "${BATCH_PER_GPU}" \
  --num_workers 8 \
  --max_epochs 100 \
  --precision 32 \
  --gradient_clip_val 1.0 \
  --t_eps 0.03 \
  --T_rev 1.0 \
  --sigma_min 0.0 \
  --sigma_max 0.487 \
  --use_mfse \
  --mf_weight_final 0.25 \
  --mf_warmup_frac 0.5 \
  --mf_delta_gamma_start 8.0 \
  --mf_delta_gamma_end 1.0 \
  --mf_delta_warmup_frac 0.7 \
  --mf_r_equals_t_prob 0.1 \
  --mf_jvp_clip 5.0 \
  --mf_jvp_eps 1e-3 \
  --mf_jvp_impl fd \
  --mf_jvp_chunk 1 \
  --mf_skip_weight_thresh 0.05 \
  --val_metrics_every_n_epochs 1 \
  --log_every_n_steps 10 \
  --default_root_dir "${LOG_DIR}" \
  --train_subset_ratio 0.3 \
  --scale_sisnr 1.0

# Notes:
# 1) Single node, 4 GPUs (Note: The script is currently set to use 2 GPUs); torchrun handles multi-processing, where each rank reads --devices 1 and binds only to its own GPU.
# 2) Both logs and checkpoints (ckpt) are saved under lightning_logs/<exp_name>/; TensorBoard reads this root directory directly.
# 3) FD-JVP + per-sample chunking + skipping MF in early curriculum stages, which balances stability and VRAM usage.
# 4) Local monitoring: run `tensorboard --logdir lightning_logs --port 6006` in your terminal, and access http://localhost:6006 in your browser.