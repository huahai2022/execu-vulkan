# SmolVLM Vulkan Device Summary

## Goal

Run `HuggingFaceTB/SmolVLM-500M-Instruct` on the connected Android phone with the Vulkan backend, and push the model execution as far as possible onto the phone GPU.

## Final result

The working route is the **split-module Vulkan path**, not the older single combined `.pte` path.

What works now:

- `vision_encoder` runs on device from `vision_encoder_vulkan.pte`
- `token_embedding` runs on device from `token_embedding_vulkan.pte`
- `text_decoder` runs on device from `text_decoder_vulkan.pte`
- the phone completes prewarm, multimodal prefill, and token generation without the earlier Vulkan runtime crash
- the runner prints TTFT, multimodal prefill speed, decode speed, and related metrics

What does not work yet:

- the generated text is still semantically wrong and collapses to repeated `<|endoftext|>` tokens

So this is a successful **systems/runtime** result, but not yet a successful **model-quality** result.

## Code map

### Files under this directory

- `export_smolvlm_vulkan.py`
  - Main custom SmolVLM exporter.
  - Loads the local Hugging Face checkpoint.
  - Builds three logical modules:
    - `SmolVLMVisionEncoder`
    - `SmolVLMTokenEmbedding`
    - `SmolVLMTextDecoder`
  - Converts the SmolVLM text weights into the ExecuTorch Llama-style text model.
  - Registers Python shims for missing local `torchao` and `llama` custom ops.
  - Exports and lowers modules through `VulkanPartitioner`.
  - Can write a combined multimethod `.pte`.

- `export_smolvlm_vulkan_split.py`
  - Preferred export path for device GPU execution.
  - Uses the same wrappers from `export_smolvlm_vulkan.py`.
  - Exports three independent Vulkan programs:
    - `vision_encoder_vulkan.pte`
    - `token_embedding_vulkan.pte`
    - `text_decoder_vulkan.pte`
  - This is the path that makes it practical to run all three model parts on device.

- `smolvlm_runner.cpp`
  - Android ExecuTorch runner for SmolVLM.
  - Supports:
    - combined multimethod `.pte`
    - split-module execution
    - precomputed image embeddings
    - precomputed prompt embeddings
    - debug modes such as `--run_vision_only`
  - Important additions:
    - `SplitTextDecoderRunner` to chain `token_embedding` and `text_decoder`
    - explicit Vulkan prewarm
    - split execution flags:
      - `--vision_model_path`
      - `--token_model_path`
      - `--text_model_path`
    - metric printing:
      - model load time
      - total inference time
      - TTFT
      - multimodal prefill speed
      - decode speed

- `prepare_image.py`
  - Converts an input image into normalized `float32`, `CHW`, `512x512` raw binary for device input.

- `dump_image_embeddings.py`
  - Runs the host-side `vision_encoder` and writes precomputed image embeddings.
  - Useful for debugging split execution and for avoiding device-side vision during earlier debugging stages.

- `dump_prompt_embeddings.py`
  - Builds full multimodal prompt embeddings on host:
    - text prefix embeddings
    - image embeddings
    - text suffix embeddings
  - Useful for debugging the text decoder in isolation.

- `build_android.sh`
  - Builds the Android ExecuTorch tree with Vulkan enabled.
  - Builds the local `smolvlm_runner` example against that Android install tree.

- `run_on_device.sh`
  - Older helper for the single combined `.pte` path.
  - Still useful as an example of basic push/run structure, but the split-module route is the better final path.

- `hf_shims/`, `et_shims/`, `py_shims/`
  - Local Python shims used to survive the environment inconsistencies in this repo and conda setup.

- `note.md`
  - Full chronological work log, experiments, and results.

### Important repo-level fixes outside this directory

These were required to get the Vulkan route working:

- `backends/vulkan/runtime/utils/VecUtils.h`
  - `make_ivec2(...)` was relaxed to accept a 1-element list and broadcast it to 2 values.
  - This fixed a real device-side Vulkan runtime crash while running the split vision module.

- `backends/vulkan/runtime/api/containers/Tensor.cpp`
  - Dynamic Vulkan resize behavior was adjusted to avoid the earlier `valid_resize is false` failure on changing shapes.

- `backends/vulkan/patterns/sdpa.py`
  - SDPA matching was made more tolerant.

- `exir/tensor.py`
  - Added safer fallback behavior around symbolic strides.

- `exir/passes/sym_shape_eval_pass.py`
  - Relaxed the symbolic-shape path to get past export failures.

## Main architecture

### Why the combined `.pte` path was not enough

The earlier combined export path reached export success, but:

- it was harder to keep `vision_encoder` on the Vulkan path
- it hit runtime shape/resize failures more easily
- it was a poor fit for proving that all major model parts were on device

### Why the split path is the correct final route

The split path makes the execution structure explicit:

1. `vision_encoder_vulkan.pte`
   - converts the image tensor into image embeddings on device
2. `token_embedding_vulkan.pte`
   - converts text token ids into embeddings on device
3. `text_decoder_vulkan.pte`
   - consumes embeddings and performs the LLM decoding on device

This is the cleanest way to say that the **model modules themselves** are on the phone and on the Vulkan route.

Important nuance:

- tokenizer logic
- CLI / runner orchestration
- sampling / token post-processing

are still CPU-side host logic inside the Android runner. So "all parts on GPU" is true for the exported model modules, not for the entire application control flow.

## Host workflow

### 1. Activate environment

```bash
cd /home/zzl/project4/execu-VULKAN/executorch
conda activate executorch-VULCAN
export PYTHONNOUSERSITE=1
```

### 2. Prepare image input

```bash
python examples/models/smolvlm_device_gpu/prepare_image.py \
  --image /absolute/path/to/your_image.jpg \
  --output examples/models/smolvlm_device_gpu/artifacts/your_image_512_chw_f32.bin
```

### 3. Export the split Vulkan modules

```bash
python examples/models/smolvlm_device_gpu/export_smolvlm_vulkan_split.py \
  --model-dir /home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --output-dir examples/models/smolvlm_device_gpu/artifacts/split_vulkan \
  --max-seq-len 256 \
  --max-context-len 256 \
  --vulkan-force-fp16 \
  --vulkan-storage buffer \
  --print-delegation-summary
```

Expected artifacts:

- `examples/models/smolvlm_device_gpu/artifacts/split_vulkan/vision_encoder_vulkan.pte`
- `examples/models/smolvlm_device_gpu/artifacts/split_vulkan/token_embedding_vulkan.pte`
- `examples/models/smolvlm_device_gpu/artifacts/split_vulkan/text_decoder_vulkan.pte`

### 4. Build Android runner

```bash
export ANDROID_NDK=/home/zzl/project/AndroidNDK26
export VULKAN_GLSLC_DIR=/home/zzl/project4/execu-VULKAN/tools/1.4.341.1/x86_64/bin

bash examples/models/smolvlm_device_gpu/build_android.sh
```

Output runner:

- `cmake-out-android-smolvlm-vulkan-min/examples/models/smolvlm_device_gpu/smolvlm_runner`

## Device workflow

### 1. Confirm device

```bash
adb devices
```

### 2. Push files

```bash
adb shell mkdir -p /data/local/tmp/smolvlm_vulkan_split_gpu_only

adb push examples/models/smolvlm_device_gpu/artifacts/split_vulkan/vision_encoder_vulkan.pte \
  /data/local/tmp/smolvlm_vulkan_split_gpu_only/vision_encoder_vulkan.pte

adb push examples/models/smolvlm_device_gpu/artifacts/split_vulkan/token_embedding_vulkan.pte \
  /data/local/tmp/smolvlm_vulkan_split_gpu_only/token_embedding_vulkan.pte

adb push examples/models/smolvlm_device_gpu/artifacts/split_vulkan/text_decoder_vulkan.pte \
  /data/local/tmp/smolvlm_vulkan_split_gpu_only/text_decoder_vulkan.pte

adb push /home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47/tokenizer.json \
  /data/local/tmp/smolvlm_vulkan_split_gpu_only/tokenizer.json

adb push examples/models/smolvlm_device_gpu/artifacts/statue_of_liberty_512_chw_f32.bin \
  /data/local/tmp/smolvlm_vulkan_split_gpu_only/image_512_chw_f32.bin

adb push cmake-out-android-smolvlm-vulkan-min/examples/models/smolvlm_device_gpu/smolvlm_runner \
  /data/local/tmp/smolvlm_vulkan_split_gpu_only/smolvlm_runner

adb shell chmod +x /data/local/tmp/smolvlm_vulkan_split_gpu_only/smolvlm_runner
```

### 3. Run on device

```bash
adb shell '/data/local/tmp/smolvlm_vulkan_split_gpu_only/smolvlm_runner \
  --vision_model_path=/data/local/tmp/smolvlm_vulkan_split_gpu_only/vision_encoder_vulkan.pte \
  --token_model_path=/data/local/tmp/smolvlm_vulkan_split_gpu_only/token_embedding_vulkan.pte \
  --text_model_path=/data/local/tmp/smolvlm_vulkan_split_gpu_only/text_decoder_vulkan.pte \
  --tokenizer_path=/data/local/tmp/smolvlm_vulkan_split_gpu_only/tokenizer.json \
  --image_bin_path=/data/local/tmp/smolvlm_vulkan_split_gpu_only/image_512_chw_f32.bin \
  --prompt="Can you describe this image?" \
  --temperature=0 \
  --max_new_tokens=8'
```

## Important operations that were done during debugging

### Environment repair

- fixed package mismatches in the `executorch-VULCAN` conda environment
- avoided `~/.local` pollution with `PYTHONNOUSERSITE=1`
- installed or restored missing Python dependencies needed for export

### Export-stage debugging

- added shim-based loading to survive missing `torchao` and custom-op shared libraries
- patched symbolic-shape behavior so `to_executorch()` could succeed
- simplified `vision_encoder` export so it no longer relied on the heavy Hugging Face patch-mask path

### Device runtime debugging

- added explicit Vulkan prewarm
- added diagnostic modes such as `--run_vision_only`
- added split execution support so vision, token embedding, and text decoding could each be run as explicit device modules
- fixed a Vulkan runtime bug in `make_ivec2(...)`
- fixed the earlier dynamic resize/runtime crash path sufficiently for split execution
- added metric printing to the custom runner

## Successful device run observations

Observed from the successful split run:

- split prewarm finished
- multimodal prefill finished
- generation loop finished
- no `vkcompute::vkapi::Error`

Observed generated output:

- repeated `<|endoftext|>` tokens

So:

- runtime path: successful
- GPU module execution: successful
- caption quality: not successful yet

## Metrics from the successful split run

- `model_load_time_ms=1339.000`
- `total_inference_time_ms=1243.000`
- `ttft_ms=930.000`
- `multimodal_prefill_time_ms=930.000`
- `multimodal_prefill_speed_tokens_per_s=87.097`
- `decode_time_ms=313.000`
- `decode_speed_tokens_per_s=22.364`
- `avg_decode_time_per_token_ms=44.714`
- `total_generated_token_throughput_tokens_per_s=5.632`
- `prompt_tokens=81`
- `generated_tokens=7`
- `sampling_time_ms=2`

## Practical conclusion

If the question is "can SmolVLM be run on the device with its major model modules on the Vulkan path?", the answer is yes through the split-module route.

If the question is "is the end-to-end model behavior already correct?", the answer is no. The remaining problem is not device Vulkan execution anymore; it is the quality/correctness of the generated output.
