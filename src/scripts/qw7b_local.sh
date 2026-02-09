#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

cd "$ROOT_DIR/src/virft"

export DEBUG_MODE=${DEBUG_MODE:-"true"}
export LOG_PATH=${LOG_PATH:-"./debug_log_am_local.txt"}

export DATA_PATH=${DATA_PATH:-"/path/to/hf_dataset_on_disk"}
export CKPT_PATH=${CKPT_PATH:-"/path/to/Qwen2.5-VL-7B-Instruct"}
export SAVE_PATH=${SAVE_PATH:-"$ROOT_DIR/share_models/Q-Hawkeye-outputs"}
export JOB_NAME=${JOB_NAME:-"Q_Hawkeye_local_perception_loss_ua_grpo"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
GPU_NUM=${GPU_NUM:-4}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-12345}

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$ROOT_DIR/.cache/triton"}
export DEEPSPEED_CACHE_DIR=${DEEPSPEED_CACHE_DIR:-"$ROOT_DIR/.cache/deepspeed"}
mkdir -p "$TRITON_CACHE_DIR" "$DEEPSPEED_CACHE_DIR" "$SAVE_PATH"

ARGS=(
  --where local
  --model_name_or_path "$CKPT_PATH"
  --output_dir "$SAVE_PATH/$JOB_NAME"
  --use_vllm false
  --do_train true
  --dataloader_num_workers 8
  --dataset_name "$DATA_PATH"
  --deepspeed local_scripts/zero3.json
  --max_prompt_length 8192
  --max_completion_length 2048
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 2
  --logging_steps 1
  --bf16
  --torch_dtype bfloat16
  --learning_rate 5e-6
  --epsilon_high 0.28
  --max_grad_norm 0.1
  --lr_scheduler_type cosine
  --warmup_steps 50
  --report_to none
  --log_completions true
  --gradient_checkpointing true
  --attn_implementation flash_attention_2
  --max_pixels 786432
  --num_train_epochs 15
  --preprocessing_num_workers 1
  --run_name "$JOB_NAME"
  --save_steps 100
  --save_total_limit 2
  --num_generations ${NUM_GENERATIONS:-4}
  --enable_perception_loss true
  --perception_loss_gamma ${PERCEPTION_LOSS_GAMMA:-0.0001}
  --perception_loss_kl_zero_epoch ${PERCEPTION_LOSS_KL_ZERO_EPOCH:-5}
  --perception_loss_eta1 ${PERCEPTION_LOSS_ETA1:-0.001}
  --perception_loss_eta2 ${PERCEPTION_LOSS_ETA2:-0.001}
  --use_ua_grpo true
  --ua_alpha ${UA_ALPHA:-0.2}
  --ua_score_min ${UA_SCORE_MIN:-1.0}
  --ua_score_max ${UA_SCORE_MAX:-5.0}
)

torchrun --nproc_per_node="$GPU_NUM" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  -- \
  src/open_r1/grpo_am.py \
  "${ARGS[@]}"
