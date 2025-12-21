#!/usr/bin/env bash
set -euo pipefail

# 基本参数（位置参数 seed，也可用环境变量覆盖）
SEED="${1:-0}"
VERSION="${VERSION:-0}"
ACTION="${ACTION:-12}"      # 2: [['right'], ['right','A']], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT
OBS="${OBS:-4}"            # 1: 不叠帧，4: 叠4帧
MAX_ENV_STEP="${MAX_ENV_STEP:-5000000}"
REWARD_MODE="${REWARD_MODE:-dense}"  # dense: 连续奖励；sparse: 仅通关/死亡
DUELING_AGGREGATOR="${DUELING_AGGREGATOR:-lse}"  # mean / max / lse；留空使用配置默认值

# 可选超参（留空即使用配置默认值）
COLLECTOR_ENV_NUM="${COLLECTOR_ENV_NUM:-16}"
EVALUATOR_ENV_NUM="${EVALUATOR_ENV_NUM:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
UPDATE_PER_COLLECT="${UPDATE_PER_COLLECT:-}"
TARGET_UPDATE_FREQ="${TARGET_UPDATE_FREQ:-}"
DISCOUNT_FACTOR="${DISCOUNT_FACTOR:-}"
NSTEP="${NSTEP:-}"
EPS_START="${EPS_START:-}"
EPS_END="${EPS_END:-}"
EPS_DECAY="${EPS_DECAY:-}"
REPLAY_BUFFER_SIZE="${REPLAY_BUFFER_SIZE:-}"
SAVE_CKPT_AFTER_ITER="${SAVE_CKPT_AFTER_ITER:-100000}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-}"

args=(
  -s "${SEED}"
  -v "${VERSION}"
  -a "${ACTION}"
  -o "${OBS}"
  --reward_mode "${REWARD_MODE}"
  --max_env_step "${MAX_ENV_STEP}"
)

# 只有设置了才追加
[[ -n "${COLLECTOR_ENV_NUM}" ]] && args+=(--collector_env_num "${COLLECTOR_ENV_NUM}")
[[ -n "${EVALUATOR_ENV_NUM}" ]] && args+=(--evaluator_env_num "${EVALUATOR_ENV_NUM}")
[[ -n "${LR}" ]] && args+=(--learning_rate "${LR}")
[[ -n "${BATCH_SIZE}" ]] && args+=(--batch_size "${BATCH_SIZE}")
[[ -n "${UPDATE_PER_COLLECT}" ]] && args+=(--update_per_collect "${UPDATE_PER_COLLECT}")
[[ -n "${TARGET_UPDATE_FREQ}" ]] && args+=(--target_update_freq "${TARGET_UPDATE_FREQ}")
[[ -n "${DISCOUNT_FACTOR}" ]] && args+=(--discount_factor "${DISCOUNT_FACTOR}")
[[ -n "${NSTEP}" ]] && args+=(--nstep "${NSTEP}")
[[ -n "${EPS_START}" ]] && args+=(--eps_start "${EPS_START}")
[[ -n "${EPS_END}" ]] && args+=(--eps_end "${EPS_END}")
[[ -n "${EPS_DECAY}" ]] && args+=(--eps_decay "${EPS_DECAY}")
[[ -n "${REPLAY_BUFFER_SIZE}" ]] && args+=(--replay_buffer_size "${REPLAY_BUFFER_SIZE}")
[[ -n "${SAVE_CKPT_AFTER_ITER}" ]] && args+=(--save_ckpt_after_iter "${SAVE_CKPT_AFTER_ITER}")
[[ -n "${TRAIN_ITERATIONS}" ]] && args+=(--train_iterations "${TRAIN_ITERATIONS}")
[[ -n "${DUELING_AGGREGATOR}" ]] && args+=(--dueling_aggregator "${DUELING_AGGREGATOR}")

echo "Running with params: ${args[*]}"
python3 -u mario_dqn_main.py "${args[@]}"