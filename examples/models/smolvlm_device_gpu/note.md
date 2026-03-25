## Goal

Run `HuggingFaceTB/SmolVLM-500M-Instruct` on the connected Android phone with the Vulkan backend only.

## What was added under this directory

- `export_smolvlm_vulkan.py`
  - Custom exporter for SmolVLM using lower-level ExecuTorch APIs.
  - Loads the Hugging Face SmolVLM checkpoint locally.
  - Splits export into `vision_encoder`, `token_embedding`, and `text_decoder`.
  - Lowers `text_decoder` through `VulkanPartitioner`.
- `prepare_image.py`
  - Converts an image into normalized `float32` CHW `512x512` raw input.
- `smolvlm_runner.cpp`
  - Multimodal runner using ExecuTorch LLM runner APIs.
- `CMakeLists.txt`
  - Android build for the runner.
- `build_android.sh`
  - Builds Android ExecuTorch artifacts and the example runner.
- `run_on_device.sh`
  - Pushes model/runtime files to the phone and executes the runner.
- `hf_shims/` and `et_shims/`
  - Local Python shims for missing `six` and `torchao` imports in this environment.

## Environment/workflow findings

- The Python environment can import `transformers`, but only through `~/.local/lib/python3.10/site-packages`, and that path is missing pieces expected by the repo.
- The stock ExecuTorch LLM export path is not usable here because `torchao` is absent and the repo checkout is also missing several third-party CMake dependencies needed to build the host-side custom-op shared library.
- Because of that, the exporter had to:
  - load HF with one shim set,
  - switch to a different shim set for ExecuTorch imports,
  - register fake `torchao` ops,
  - register Python `llama` custom-op shims so export can proceed without `libcustom_ops_aot_lib.so`.

## Additional repo fixes made during debugging

- `backends/vulkan/patterns/sdpa.py`
  - Made Vulkan SDPA fusion more tolerant of cache-view wrappers and incomplete matches.
- `exir/tensor.py`
  - Added safer fallback behavior when tensor dim-order inference hits data-dependent symbolic strides.
- `exir/passes/sym_shape_eval_pass.py`
  - Relaxed hint-based symbolic-shape handling so it does not immediately assert on unbounded hinted ranges.

## Verified intermediate results

- Image preprocessing works:
  - `prepare_image.py` successfully wrote a `float32` `3x512x512` raw tensor.
- The custom exporter script parses and starts successfully.
- The export now gets through:
  - HF model load,
  - text-model construction,
  - Vulkan partitioning.

## Export result

The export blocker was resolved.

- Root cause of the old `to_executorch()` failure was the SmolVLM vision path creating `torch.arange()` from symbolic patch counts.
- `export_smolvlm_vulkan.py` was rewritten so `SmolVLMVisionEncoder` uses a fixed patch grid, fixed `position_ids`, and a fixed full patch-attention mask for the `512x512` image case.
- The exporter now writes a working multimethod `.pte`:
  - `examples/models/smolvlm_device_gpu/artifacts/smolvlm_vulkan_gpu.pte`
- The sample export inputs were also enlarged to `max_context_len`, fixing the earlier bounded-input runtime error (`maximum capacity of 4 elements to 5 elements`).

## Device run attempts after export

The Android runner now builds and the model loads on device. The remaining blocker moved to the Vulkan runtime on the phone.

Observed device behavior:

- The runner binary builds successfully under:
  - `cmake-out-android-smolvlm-vulkan-min/examples/models/smolvlm_device_gpu/smolvlm_runner`
- The `.pte`, tokenizer, image tensor, and runner push to the phone successfully.
- The process starts, loads metadata, and begins executing on device.

## Vulkan runtime investigation

The original on-device failure was:

- `vkcompute::vkapi::Error`
- `check_sizes ... (valid_resize) is false! tensor sizes requires a larger texture than the current one`

This means Vulkan delegate/runtime textures are being allocated for one dynamic execution shape and then a later execution requests a different shape family that cannot reuse or grow the existing texture.

### What was tried

- Reworked `smolvlm_runner.cpp` so it no longer uses the opaque factory directly:
  - it builds the multimodal runtime stack explicitly,
  - uses `Module::LoadMode::MmapUseMlockIgnoreErrors`,
  - and performs explicit Vulkan prewarm on the same module instance.
- Added staged prewarm experiments for:
  - `token_embedding`
  - `text_decoder` with long token embeddings
  - `vision_encoder`
  - `text_decoder` with image embeddings
- Tried several prewarm shape/order strategies:
  - token prefill first, then decode
  - decode first, then prefill
  - one synthetic high-start text-decoder shape
  - image-embedding prewarm padded to the practical target length
  - combined token-embedding and image-embedding prewarm
- Added a custom `CloningMultimodalPrefiller` in the example runner so both:
  - token embeddings
  - image embeddings
  are cloned to owned tensors before being fed to `text_decoder`, to remove method-output alias/layout differences.
- Updated prewarm to use the same `populate_start_pos_or_cache_position()` helper as the real runtime path so the cache-position tensor family matches actual execution more closely.

### Latest device state

The explicit prewarm now completes much further than before:

- `token_embedding(len=96)` succeeds
- `vision_encoder` succeeds
- `text_decoder(image embeddings padded_len=96, start_pos=95)` succeeds
- `text_decoder(token embeddings len=96, start_pos=95)` succeeds
- `Vulkan prewarm finished` is printed on device

But the real multimodal generation still aborts immediately afterward with the same Vulkan resize error:

- log sequence:
  - `Vulkan prewarm finished`
  - `RSS after loading model: ...`
  - then `vkcompute::vkapi::Error ... valid_resize is false`

So the export is complete, but the final remaining blocker is still a Vulkan runtime shape/allocation incompatibility in the actual multimodal execution path on device.

## Current practical conclusion

Status at the end of these attempts:

- `.pte` export: finished
- Android runner build: finished
- push to phone: finished
- on-device GPU execution: still blocked by Vulkan runtime texture-resize failure

At this point the next step is deeper Vulkan/runtime debugging, not more export work. The example runner already contains the current best local workaround attempts under this directory.

## 2026-03-22 device rerun after exporter fix

The Python/export environment was repaired and `smolvlm_vulkan_gpu.pte` now exports successfully in:

- `torch 2.11.0+cpu`
- `transformers 5.0.0rc1`
- `huggingface_hub 1.7.2`
- `tokenizers 0.22.2`
- `accelerate 1.13.0`
- `sympy 1.14.0`
- `ruamel.yaml 0.19.1`

The exact export command now works:

```bash
export PYTHONNOUSERSITE=1
python examples/models/smolvlm_device_gpu/export_smolvlm_vulkan.py \
  --model-dir /home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --output examples/models/smolvlm_device_gpu/artifacts/smolvlm_vulkan_gpu.pte \
  --max-seq-len 256 \
  --max-context-len 256 \
  --vulkan-force-fp16
```

### Fresh device push

The following files were pushed again to:

- `/data/local/tmp/smolvlm_vulkan/smolvlm_vulkan.pte`
- `/data/local/tmp/smolvlm_vulkan/tokenizer.json`
- `/data/local/tmp/smolvlm_vulkan/image_512_chw_f32.bin`
- `/data/local/tmp/smolvlm_vulkan/smolvlm_runner`

The `.pte` push completed successfully at about `17.7 MB/s`.

### Full on-device run attempt

Command run:

```bash
adb shell '/data/local/tmp/smolvlm_vulkan/smolvlm_runner \
  --model_path=/data/local/tmp/smolvlm_vulkan/smolvlm_vulkan.pte \
  --tokenizer_path=/data/local/tmp/smolvlm_vulkan/tokenizer.json \
  --image_bin_path=/data/local/tmp/smolvlm_vulkan/image_512_chw_f32.bin \
  --prompt="Can you describe this image?" \
  --temperature=0 \
  --max_new_tokens=16 \
  --prewarm_context_len=96'
```

Observed output:

- metadata loads successfully
- `Prewarming Vulkan methods with prewarm_context_len=96`
- `Prewarm step: token_embedding(len=96)`
- `Prewarm step: vision_encoder`

After that, no new runner log lines appeared for multiple minutes.

Process inspection while stalled:

- `smolvlm_runner` remained alive in `R` state
- one thread stayed near `98%` CPU
- RSS grew to roughly `427 MB`
- it did not advance to `text_decoder(...)`

This means the device run is currently spending a very long time inside `vision_encoder`, not immediately crashing.

### Diagnostic runner changes

Two temporary diagnostic flags were added to `smolvlm_runner.cpp`:

- `--skip_prewarm`
- `--run_vision_only`

These were added only to isolate whether the problem is in prewarm orchestration or in the `vision_encoder` method itself.

### Isolated vision-encoder test

Command run:

```bash
adb shell '/data/local/tmp/smolvlm_vulkan/smolvlm_runner \
  --model_path=/data/local/tmp/smolvlm_vulkan/smolvlm_vulkan.pte \
  --run_vision_only=1'
```

Observed output:

- `Diagnostic: running vision_encoder only`

After that, it again stayed active for more than a minute without returning.

Process inspection during this diagnostic:

- process state remained `R`
- one thread stayed around `97.8%` CPU
- RSS reached about `245 MB`

### Single-thread vision diagnostic

Command run:

```bash
adb shell '/data/local/tmp/smolvlm_vulkan/smolvlm_runner \
  --model_path=/data/local/tmp/smolvlm_vulkan/smolvlm_vulkan.pte \
  --run_vision_only=1 \
  --cpu_threads=1'
```

Observed output:

- `Resetting threadpool to 1 threads.`
- `Diagnostic: running vision_encoder only`

It still failed to return promptly, so the current issue is not caused by the threadpool width.

## Current conclusion after the latest rerun

The device execution blocker has shifted again:

- export now works
- `.pte` pushes correctly
- the Android runner launches correctly

## 2026-03-22 vision-encoder deep dive and final GPU-only route

The important conclusion from the later investigation is that the old single-file multimethod `.pte` is not the right route if the goal is "all model parts on device GPU". The correct route is the split-module Vulkan export:

- `vision_encoder_vulkan.pte`
- `token_embedding_vulkan.pte`
- `text_decoder_vulkan.pte`

### What was verified about `vision_encoder`

- The simplified `SmolVLMVisionEncoder` wrapper in `export_smolvlm_vulkan.py` now bypasses the Hugging Face patch-mask construction path and uses:
  - fixed `position_ids`
  - direct patch embedding
  - direct encoder call with `attention_mask=None`
  - fixed `512x512` image assumptions
- Exporting `vision_encoder` alone and lowering it through `VulkanPartitioner` now produces:
  - `Delegation summary: vision_encoder: delegates=1 fully_delegated=1`
- So the `vision_encoder` graph itself is now fully delegated to Vulkan when exported as its own module.

This matters because the earlier "vision has remaining `aten.any/eq/ge/index/logical_not/...` ops" conclusion came from an older mixed state during investigation. The current isolated `vision_encoder` graph is fully delegated.

### Practical GPU-only path

For this example, "all model parts on GPU" means:

- `vision_encoder` runs on device from `vision_encoder_vulkan.pte`
- `token_embedding` runs on device from `token_embedding_vulkan.pte`
- `text_decoder` runs on device from `text_decoder_vulkan.pte`

This is the path exercised by the split runner flags:

- `--vision_model_path`
- `--token_model_path`
- `--text_model_path`

Tokenizer handling, CLI parsing, and general runner control flow still run on CPU, but the model modules themselves are on the split Vulkan route.

### Fresh on-device split run

Files pushed to:

- `/data/local/tmp/smolvlm_vulkan_split_gpu_only/vision_encoder_vulkan.pte`
- `/data/local/tmp/smolvlm_vulkan_split_gpu_only/token_embedding_vulkan.pte`
- `/data/local/tmp/smolvlm_vulkan_split_gpu_only/text_decoder_vulkan.pte`
- `/data/local/tmp/smolvlm_vulkan_split_gpu_only/tokenizer.json`
- `/data/local/tmp/smolvlm_vulkan_split_gpu_only/image_512_chw_f32.bin`
- `/data/local/tmp/smolvlm_vulkan_split_gpu_only/smolvlm_runner`

Command run:

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

Observed device result:

- split prewarm completed successfully
  - `token_embedding(len=96)`
  - `text_decoder(token embeddings len=96 start_pos=95)`
  - `vision_encoder`
  - `text_decoder(image embeddings padded_len=96 start_pos=95)`
- runtime multimodal prefill completed
  - text prefix prefill
  - image-embedding prefill
  - text suffix prefill
- generation completed without Vulkan runtime failure

Observed output tokens:

- repeated `<|endoftext|>` tokens

So the final runtime state is:

- on-device split execution: works
- Vulkan runtime crash: fixed for this split path
- all three model modules on phone: works
- useful caption quality: still not solved

### Metrics from the successful split device run

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

### Final experiment result

The "run SmolVLM on phone using GPU for all model parts" goal is achieved through the split-module Vulkan path, not the older combined `.pte` path.

The remaining unsolved issue is output correctness:

- runtime is stable
- model modules execute on device through the split Vulkan flow
- generation still collapses to repeated end-of-text tokens

So this is a successful systems/runtime result, but not yet a correct model-quality result.
- but the on-device run is presently bottlenecked in `vision_encoder`

So, for the current repo state on this phone:

- the Vulkan text-decoder path is not the first blocker reached anymore
- the first blocker reached in practice is very slow or effectively non-terminating `vision_encoder` execution on device

The next step should be one of:

- isolate and benchmark the `vision_encoder` graph separately with a smaller CPU-only harness
- inspect whether `vision_encoder` is falling back to an unexpectedly slow kernel path on Android
- or export/replace the vision path with a simpler or precomputed embedding workflow if the goal is specifically to validate the Vulkan text-decoder delegate on device

## 2026-03-22 follow-up: skip image prewarm and retry full generation

To test whether the earlier Vulkan runtime failure still exists once image prewarm is removed, `smolvlm_runner.cpp` was updated with:

- `--skip_image_prewarm`

This flag keeps:

- `token_embedding` prewarm
- `text_decoder(token embeddings ...)` prewarm

but skips:

- `vision_encoder` prewarm
- `text_decoder(image embeddings ...)` prewarm

### Command run

```bash
adb shell '/data/local/tmp/smolvlm_vulkan/smolvlm_runner \
  --model_path=/data/local/tmp/smolvlm_vulkan/smolvlm_vulkan.pte \
  --tokenizer_path=/data/local/tmp/smolvlm_vulkan/tokenizer.json \
  --image_bin_path=/data/local/tmp/smolvlm_vulkan/image_512_chw_f32.bin \
  --prompt="Can you describe this image?" \
  --temperature=0 \
  --max_new_tokens=16 \
  --prewarm_context_len=96 \
  --skip_image_prewarm=1'
```

### Observed behavior

This time the runner no longer stalled in prewarm. It printed:

- `Prewarm step: token_embedding(len=96)`
- `Skipping image-related prewarm by request`
- `Prewarm step: text_decoder(token embeddings len=96 start_pos=95)`
- `Vulkan prewarm finished`

Then execution advanced into the real runtime path and immediately failed with the Vulkan error that had been seen in earlier experiments:

- `Runtime prefill: text/tokens len=5 start_pos=0`
- `Runtime prefill: text_decoder text/tokens seq_len=5 start_pos=0`
- `vkcompute::vkapi::Error`
- `check_sizes ... (valid_resize) is false! tensor sizes requires a larger texture than the current one`

### Updated conclusion

Yes, removing image prewarm makes the run reach the later Vulkan failure again.

That means there are at least two separate device-side issues in this setup:

1. `vision_encoder` can take impractically long when prewarmed directly on device.
2. Even if image prewarm is skipped, the Vulkan-lowered `text_decoder` still hits the known dynamic texture-resize failure during actual generation.

So the later Vulkan failure is still real and reproducible; the image-prewarm stall was only masking it in the more recent runs.

## 2026-03-22 final follow-up: Vulkan resize bug fixed enough to run on device

The remaining device blocker turned out not to be purely model-specific. The Vulkan runtime tensor resize path had a real logic bug.

### What was changed

#### Host-side helpers added

New helper scripts were added under `examples/models/smolvlm_device_gpu/`:

- `dump_image_embeddings.py`
  - loads the HF SmolVLM checkpoint
  - runs the fixed `SmolVLMVisionEncoder`
  - writes raw float32 image embeddings `[1, seq_len, 960]`
- `dump_prompt_embeddings.py`
  - loads the HF checkpoint
  - builds a full prompt-embedding tensor by concatenating:
    - prefix text embeddings
    - image embeddings
    - suffix text embeddings
  - writes raw float32 prompt embeddings `[1, seq_len, 960]`

#### Runner changes

`smolvlm_runner.cpp` now supports:

- `--image_embeddings_bin_path`
- `--prompt_embeddings_bin_path`

This makes it possible to bypass on-device `vision_encoder` entirely and run:

- text prefill + precomputed image embeddings + suffix text prefill, or
- one-shot prefill from a fully precomputed prompt embedding tensor

#### Vulkan runtime fix

The critical runtime changes were in:

- `backends/vulkan/runtime/api/containers/Tensor.cpp`

Two issues were fixed:

1. `check_sizes()` was validating resizes using the tensor's old `padded_sizes_` instead of the requested new sizes.
2. `virtual_resize()` for texture-backed tensors could not recover when a legal dynamic shape required a different underlying image extent layout.

The updated path now:

- computes validation extents from the requested size
- reallocates texture-backed storage when the new extents exceed the current backing image

This was the key reason the earlier dynamic Vulkan runs kept aborting with:

- `check_sizes ... (valid_resize) is false!`

### Important build note

The Android example was linking against:

- `cmake-out-android-smolvlm-vulkan-min/lib/libvulkan_backend.a`

but the freshly rebuilt archive was written to:

- `cmake-out-android-smolvlm-vulkan-min/backends/vulkan/libvulkan_backend.a`

So after rebuilding the Vulkan backend, the runner had to be relinked against the updated backend archive.

### Commands run successfully

#### 1. Generate image embeddings on host

```bash
export PYTHONNOUSERSITE=1
/home/zzl/tools/conda/envs/executorch-VULCAN/bin/python \
  examples/models/smolvlm_device_gpu/dump_image_embeddings.py \
  --model-dir /home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --image-bin examples/models/smolvlm_device_gpu/artifacts/statue_of_liberty_512_chw_f32.bin \
  --output examples/models/smolvlm_device_gpu/artifacts/statue_of_liberty_image_embeddings.bin
```

Observed result:

- wrote `shape=(1, 64, 960)`

#### 2. Generate prompt embeddings on host

```bash
export PYTHONNOUSERSITE=1
/home/zzl/tools/conda/envs/executorch-VULCAN/bin/python \
  examples/models/smolvlm_device_gpu/dump_prompt_embeddings.py \
  --model-dir /home/zzl/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-500M-Instruct/snapshots/a7da5b986cb59b408707209984f360a5f4ad7e47 \
  --image-bin examples/models/smolvlm_device_gpu/artifacts/statue_of_liberty_512_chw_f32.bin \
  --output examples/models/smolvlm_device_gpu/artifacts/statue_of_liberty_prompt_embeddings.bin \
  --prompt "Can you describe this image?"
```

Observed result:

- wrote `shape=(1, 81, 960)`
- `prefix_len=5`
- `image_len=64`
- `suffix_len=12`

#### 3. Push the updated runner and artifacts

```bash
adb push cmake-out-android-smolvlm-vulkan-min/examples/models/smolvlm_device_gpu/smolvlm_runner /data/local/tmp/smolvlm_vulkan/smolvlm_runner
adb push examples/models/smolvlm_device_gpu/artifacts/statue_of_liberty_image_embeddings.bin /data/local/tmp/smolvlm_vulkan/statue_of_liberty_image_embeddings.bin
adb push examples/models/smolvlm_device_gpu/artifacts/statue_of_liberty_prompt_embeddings.bin /data/local/tmp/smolvlm_vulkan/statue_of_liberty_prompt_embeddings.bin
```

#### 4. Successful device run with prompt embeddings

Command:

```bash
adb shell '/data/local/tmp/smolvlm_vulkan/smolvlm_runner \
  --model_path=/data/local/tmp/smolvlm_vulkan/smolvlm_vulkan_buffer_fp16.pte \
  --tokenizer_path=/data/local/tmp/smolvlm_vulkan/tokenizer.json \
  --prompt_embeddings_bin_path=/data/local/tmp/smolvlm_vulkan/statue_of_liberty_prompt_embeddings.bin \
  --temperature=0 \
  --max_new_tokens=8'
```

Observed behavior:

- prewarm completed successfully
- runtime prefill completed successfully
- generation loop completed successfully
- process exited with code `0`

Observed output:

- repeated `<|endoftext|>` pieces

#### 5. Successful device run with image embeddings

Command:

```bash
adb shell '/data/local/tmp/smolvlm_vulkan/smolvlm_runner \
  --model_path=/data/local/tmp/smolvlm_vulkan/smolvlm_vulkan_buffer_fp16.pte \
  --tokenizer_path=/data/local/tmp/smolvlm_vulkan/tokenizer.json \
  --image_embeddings_bin_path=/data/local/tmp/smolvlm_vulkan/statue_of_liberty_image_embeddings.bin \
  --prompt="Can you describe this image?" \
  --temperature=0 \
  --max_new_tokens=8'
```

Observed behavior:

- custom prewarm completed successfully
- runtime prefix/image/suffix prefill completed successfully
- generation loop completed successfully
- process exited with code `0`

Observed output:

- repeated `<|endoftext|>` pieces

### Final result

The original hard Vulkan runtime failure on device was fixed enough to get the split SmolVLM path running under the Vulkan delegate on the phone.

What now works:

- export to `.pte`
- Android build
- push to device
- Vulkan text-decoder execution on device without the previous `valid_resize` abort
- full runner execution to normal process exit when using precomputed image embeddings or precomputed prompt embeddings

What still does **not** work correctly:

- output quality is not correct yet
- the current split pipeline degenerates into repeated `<|endoftext|>` generation

So the final state is:

- device Vulkan execution is working
- but the current SmolVLM split/decomposition is still not semantically faithful enough to produce a useful caption

This means the remaining problem is no longer the Android/Vulkan runtime crash. It is now model fidelity / prompt-merging correctness in the custom split pipeline.
