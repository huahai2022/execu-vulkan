#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zzl/project4/execu-VULKAN/executorch"
EXAMPLE_DIR="${ROOT_DIR}/examples/models/smolvlm_device_gpu"
BUILD_DIR="${ROOT_DIR}/cmake-out-android-smolvlm-vulkan-min"
DEVICE_DIR="/data/local/tmp/smolvlm_vulkan"

MODEL_PTE="${1:-${EXAMPLE_DIR}/artifacts/smolvlm_vulkan_tutorial_style.pte}"
IMAGE_BIN="${2:-${EXAMPLE_DIR}/artifacts/statue_of_liberty_512_chw_f32.bin}"
PROMPT="${3:-Can you describe this image?}"

TOKENIZER_JSON="/home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47/tokenizer.json"
RUNNER_BIN="${BUILD_DIR}/examples/models/smolvlm_device_gpu/smolvlm_runner"

adb shell mkdir -p "${DEVICE_DIR}"
adb push "${MODEL_PTE}" "${DEVICE_DIR}/smolvlm_vulkan.pte"
adb push "${TOKENIZER_JSON}" "${DEVICE_DIR}/tokenizer.json"
adb push "${IMAGE_BIN}" "${DEVICE_DIR}/image_512_chw_f32.bin"
adb push "${RUNNER_BIN}" "${DEVICE_DIR}/smolvlm_runner"

adb shell "chmod +x ${DEVICE_DIR}/smolvlm_runner"
adb shell "${DEVICE_DIR}/smolvlm_runner \
  --model_path=${DEVICE_DIR}/smolvlm_vulkan.pte \
  --tokenizer_path=${DEVICE_DIR}/tokenizer.json \
  --image_bin_path=${DEVICE_DIR}/image_512_chw_f32.bin \
  --prompt='${PROMPT}' \
  --temperature=0 \
  --max_new_tokens=96"
