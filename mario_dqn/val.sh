#!/usr/bin/env bash
set -euo pipefail

CKPT="${CKPT:-${1:-exp/v0_12a_4f_seed0_251220_192557/ckpt/ckpt_best.pth.tar}}"
SEED="${SEED:-${2:-0}}"
VERSION="${VERSION:-${3:-0}}"
ACTION="${ACTION:-${4:-12}}"   # 2: [['right'], ['right','A']], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT
OBS="${OBS:-${5:-4}}"         # 1: 不叠帧，4: 叠4帧
REPLAY_PATH="${REPLAY_PATH:-${6:-./eval_videos/12_21_dueling_lse}}"
REWARD_MODE="${REWARD_MODE:-dense}"  # dense: 连续奖励；sparse: 仅通关/死亡
DUELING_AGGREGATOR="${DUELING_AGGREGATOR:-lse}"  # 评估时与训练保持一致，留空则使用默认 mean

echo "Evaluating ckpt=${CKPT}, seed=${SEED}, version=${VERSION}, action=${ACTION}, obs=${OBS}, reward_mode=${REWARD_MODE}, replay_path=${REPLAY_PATH}, dueling_aggregator=${DUELING_AGGREGATOR:-default}"
extra_args=()
[[ -n "${DUELING_AGGREGATOR}" ]] && extra_args+=(--dueling_aggregator "${DUELING_AGGREGATOR}")

python3 -u evaluate.py -ckpt "${CKPT}" -s "${SEED}" -v "${VERSION}" -a "${ACTION}" -o "${OBS}" --reward_mode "${REWARD_MODE}" -rp "${REPLAY_PATH}" "${extra_args[@]}"