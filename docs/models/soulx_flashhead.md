# SoulX-FlashHead

> [!IMPORTANT]
> **cuDNN is strongly recommended for Lite on CUDA:** about **2× faster warm
> inference** with roughly half the peak VRAM in the RTX 5090 Q8 measurement
> (**9,152 → 5,168 MiB; 44% less VRAM**).
> **Prebuilt binaries do not include cuDNN support.** Build from source with
> cuDNN and `-DGGML_CUDA_CUDNN_CONV3D=ON`, then select
> `--session-option 'soulx_flashhead.exec_plan={"vae":{"impl":"cudnn"}}'`.

SoulX-FlashHead generates portrait video from a reference image and speech audio.
The family uses a shared audio-conditioned DiT with variant-specific video VAEs:
Lite uses LTX and Pro uses Wan. Only Lite GGUF weights are provided.

Upstream: [Soul-AILab/SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead).

## Download Lite

GGUF packages: [audio-cpp/SoulX-FlashHead-GGUF](https://huggingface.co/audio-cpp/SoulX-FlashHead-GGUF).
Q8 is the default package; BF16 is also available. Both include all required weights and configuration.

```bash
hf download audio-cpp/SoulX-FlashHead-GGUF SoulX-FlashHead-Lite-Q8_0.gguf \
  --local-dir models/SoulX-FlashHead-GGUF
```

## Generate

```bash
build/debug/bin/audiocpp_cli \
  --family soulx_flashhead --task gen \
  --model /path/to/SoulX-FlashHead-Lite-Q8_0.gguf \
  --backend cuda --threads 8 \
  --audio /path/to/speech.wav \
  --request-option reference_image_file=/path/to/portrait.png \
  --out-dir outputs/flashhead \
  --log --log-file outputs/flashhead/run.log
```

Use `--mode streaming` to supply audio incrementally. Streaming input must be
mono 16 kHz. Lite emits 24 frames per audio block at 25 FPS, carrying nine motion
frames between blocks. The audio encoder uses an eight-second sliding window.
A partial final audio block is padded with silence, matching upstream.

Output artifacts contain RGB24 frames with width, height, frame count, frame rate
and chunk index metadata. Streaming consumers can display each chunk as it arrives.

Encode the artifacts as an MP4, optionally muxing the source speech:

```bash
python tools/models/soulx_flashhead/render_video.py \
  --input outputs/flashhead --audio /path/to/speech.wav \
  --output outputs/flashhead/video.mp4
```

## Model

| Field | Value |
|---|---|
| Family | `soulx_flashhead` |
| Tasks | `sfx` |
| Modes | `offline`, `streaming` |
| Languages | `auto` |
| Default package | `soulx_flashhead_lite_q8_0` |
| Input | Reference portrait and speech audio |
| Output | Portrait video frames; see MP4 export instructions |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `reference_image_file` | `path` | required | Portrait image used to condition appearance. |
| `height` | `int` | `512` | Output height, divisible by the VAE spatial stride. Declared range: minimum 32. |
| `width` | `int` | `512` | Output width, divisible by the VAE spatial stride. Declared range: minimum 32. |
| `num_inference_steps` | `int` | `4` | Distilled denoising steps. Declared range: minimum 1. |
| `seed` | `int` | `42` | Generation seed. Declared range: minimum 0. |
| `color_correction_strength` | `float` | `1` | Blend output color statistics toward the reference portrait. Declared range: minimum 0, maximum 1. |
| `audio_guidance_scale` | `float` | `3` | Audio CFG scale for non-distilled pretrained checkpoints; distilled Lite/Pro do not use CFG. Declared range: minimum 0. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `soulx_flashhead.exec_plan` | `string` | `{}` | JSON component execution plan. vae.impl is ggml (default) or cudnn. cudnn requires the Lite variant, CUDA backend, and cuDNN Conv3D build support. color.impl is graph (default) or fused. fused keeps color correction on. It supports cpu and metal only and is not bit-identical on Metal. No artifacts or params are used. |

## Convert Lite

Download `Model_Lite/` and `VAE_LTX/` from
[Soul-AILab/SoulX-FlashHead-1_3B](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B),
and the model and configuration files from
[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h).

```bash
python tools/models/soulx_flashhead/convert_gguf.py \
  --source /path/to/SoulX-FlashHead-1_3B \
  --audio-encoder /path/to/wav2vec2-base-960h \
  --type bf16 \
  --output /path/to/SoulX-FlashHead-GGUF/SoulX-FlashHead-Lite-BF16.gguf
```

The GGUF contains the DiT, video VAE, speech encoder, configuration and model
contract. No Python runtime or original checkpoint files are needed for C++ inference.

## Variant boundaries

The DiT reads channel counts, transformer dimensions, patch size and temporal/spatial
VAE strides from the embedded configuration. Lite selects the LTX VAE; Pro and the
non-distilled architecture select the shared Wan VAE modules. Non-distilled weights
use Euler updates with `audio_guidance_scale` (default 3); distilled checkpoints use
the upstream re-noising update. Only Lite GGUFs are provided. Pro has also been executed from normalized source
weights. Its Wan VAE passes the forced-latent RGB comparison against upstream.
The unreleased non-distilled checkpoint has not been validated and is not an
advertised package.
The converter accepts `--variant lite|pro|pretrained`; its default is Lite.
Use `--prepare-only` with an output directory to normalize a source checkpoint
without creating a GGUF. The directory can be loaded directly by this family:

```bash
python tools/models/soulx_flashhead/convert_gguf.py \
  --source /path/to/SoulX-FlashHead-1_3B \
  --audio-encoder /path/to/wav2vec2-base-960h \
  --variant pro --prepare-only --output /path/to/FlashHead-Pro-normalized
```

Use that directory as `--model` in the generation command above.

Lite uses the session-owned LTX video VAE runtime, which caches its encode/decode
graphs and shares their scratch arena. Video stays on the backend through color
correction and motion encoding; RGB frames are copied to the host for delivery.
CUDA attention uses SageAttention2 when the build and device support its head
dimensions, otherwise Flash Attention.

Enable the optional native cuDNN Conv3D path for Lite CUDA builds:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug \
  -DENGINE_ENABLE_CUDA=ON -DGGML_CUDA_CUDNN_CONV3D=ON \
  -DCUDNN_ROOT=/path/to/cudnn
cmake --build build/debug --parallel 8
```

When explicitly selected through the execution plan below, this requires cuDNN
headers and libraries. Native Lite convolutions use F16 operands,
F32 accumulation, and F16 output storage; channel bias is added in F32 to the
F16 convolution result before the final F16 cast. Residual storage is also F16.
Normalization accumulates in F32 with F16 output on the native path. DiT
intermediates, color processing, and public VAE inputs/outputs use F32. These paths avoid reproducing every Python BF16 intermediate rounding;
small numerical differences are expected.

## Server streaming

Configure a streaming entry and start `audiocpp_server` with that config:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "threads": 8,
  "lazy_load": true,
  "models": [{
    "id": "flashhead",
    "family": "soulx_flashhead",
    "path": "/path/to/SoulX-FlashHead-Lite-Q8_0.gguf",
    "task": "gen",
    "mode": "streaming"
  }]
}
```

```bash
curl -N http://127.0.0.1:8080/v1/tasks/stream \
  -H 'Content-Type: application/json' \
  -d '{"model":"flashhead","stream_format":"sse","include_artifacts":true,"request":{"audio":"/path/to/speech-16k-mono.wav","options":{"reference_image_file":"/path/to/portrait.png"}}}'
```

Each `task.stream.event` carries video chunks in `event.artifacts`. Decode the
artifact's Base64 `payload` as RGB24 using its width, height, frames and FPS
metadata. `chunk_index` gives the order. Chunks arrive during generation;
`task.stream.done` ends the request and does not repeat their payloads. For a
complete non-streaming result, configure an `offline` entry and use
`/v1/tasks/run` instead. A streaming entry delivers video through stream events;
`/v1/tasks/run` does not collect those events into its response.

Select the generic VAE with an omitted execution plan or explicitly with
`--session-option 'soulx_flashhead.exec_plan={"vae":{"impl":"ggml"}}'`.
For Lite on CUDA, select the optional cuDNN implementation with
`--session-option 'soulx_flashhead.exec_plan={"vae":{"impl":"cudnn"}}'`.
This requires the cuDNN Conv3D build option above; an unavailable implementation
is rejected rather than silently substituted.

## Fused color correction

The default color path is unchanged: `color.impl` omitted or `"graph"` builds the
existing Lab-transfer graph. Turning `color_correction_strength` off is not an
optimization. The opt-in fused path keeps correction on and evaluates the same
transfer (bf16-rounded matrices and white point, the same sRGB and Lab knees,
per-frame mean and standard deviation, then the strength blend):

```bash
--session-option 'soulx_flashhead.exec_plan={"color":{"impl":"fused"}}'
```

`fused` is rejected on CUDA, Vulkan, and HIP. On Metal it requires a macOS build
with `ENGINE_ENABLE_METAL=ON`. Omit the field, or set `"impl":"graph"`, to roll
back. No other request option changes.

### CPU and Metal precision

Metal Shading Language 4.1, section 2.1, does not include `double`, `long double`,
or any 64-bit floating-point type. The fused Metal shader uses `float` only.
On a macOS 15 SDK the host sets `MTLMathModeSafe` and
`MTLMathFloatingPointFunctionsPrecise`. That is the MSL 1.6.3 meaning of
`-fno-fast-math`. The deprecated `fastMathEnabled` property is left unset on
that SDK. The shader source also requests `math_mode(safe)` and `contract(off)`.
An older SDK falls back to `fastMathEnabled = NO`.

The CPU fused path still reduces each Lab row with a double sum, matching
`ggml_mean` on CPU. That path was bitwise identical to the graph in the CPU run
below. The Metal stats kernel does not call that routine and does not read the
Lab image back to the host. Each row is reduced on the GPU by 256 Kahan lanes,
the lane residual is kept, and the lanes are combined with Knuth two-sum. The
same sequence is `color_fused_metal_f32_row`. A CPU run of that replica against
the double graph was bitwise identical on the test images, including constant
frames and a 5×7 frame. A 384² row of random values in about [-40, 40] differed
in the mean by `3.7e-9` and matched the double standard deviation. Full frames
using that reduction and the host per-pixel math differed from the graph by
max abs `4.6e-6` at 128 (RMSE `1.1e-8`, no 8-bit mismatches) and by max abs 0
at 384 and 512. Per-pixel
`log`, `exp`, and `sqrt` in the shader are MSL functions. Their difference from
host libm was not measured, because this host has no `metal` compiler and no
Metal device.

`match()` always runs the CPU double/libm kernel. `match_device()` on Metal runs
the shader. The test used to require those two buffers to be bitwise identical
and exited with no delta when they were not. It now prints max abs, RMSE,
non-finite counts, and the first sample, and scores that pair with the same
max abs `1e-4` and RMSE `2e-5` gate. On CPU the device and host kernels are the
same code and must still match bitwise. A Metal device-versus-graph comparison
is a separate line. The M5 run of `ce2051be` printed max abs 0 for the 1×1
device-versus-graph tail before the bitwise host check stopped the process.
384 and 512 on that GPU were not reached.

This host cannot compile the shader. `metal` and `xcrun` are absent. The shader
source was checked by reading it: the embedded MSL uses `float` and does not
declare a 64-bit float. That is not a GPU execution.

Stage timings are written to the normal timing log when logging is enabled:

| Key | Meaning |
|---|---|
| `soulx_flashhead.color.impl` | `graph`, `fused_cpu`, `fused_metal`, or `fused_copy` (strength 0) |
| `soulx_flashhead.color.lab_ms` | Fused RGB to Lab, including a cached reference on a miss |
| `soulx_flashhead.color.stats_ms` | Per-frame mean and standard deviation |
| `soulx_flashhead.color.match_ms` | Lab match, Lab to RGB, and the strength blend |
| `soulx_flashhead.color.reference_cached` | `1` when the portrait statistics were reused |
| `soulx_flashhead.color.wall_ms` | Whole color stage. The graph path reports this too |

The reference portrait statistics are computed once per identical portrait and
resolution. Video frames are still corrected every chunk. The Lite VAE decode
and motion encode are unchanged: decode is LTX Conv3D through ggml im2col and
matmul, with a bf16 round after each convolution and residual. Dropping those
rounds or switching activations to float16 would change values. This change
does not do that.

### Portable numerical check

No model weights are required. From a CPU build:

```bash
cmake -S . -B build/color-opt -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=soulx_flashhead \
  -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_METAL=OFF \
  -DENGINE_ENABLE_NATIVE_CPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/color-opt -j"$(nproc)" --target soulx_flashhead_color_opt_test
./build/color-opt/bin/soulx_flashhead_color_opt_test
./build/color-opt/bin/soulx_flashhead_color_opt_test --bench 128
```

The test compares the graph and the fused kernel on edge values (sRGB knee,
out-of-range samples, a constant frame, a one-pixel spike), non-square sizes,
strength 0 / 0.37 / 1, and motion tails 9 and 5. Strength 0, a repeated fused
call, and thread counts 1 and 2 are bitwise identical. For strength greater
than 0 the regression gate is:

- max absolute error `1e-4` on the `[-1, 1]` samples
- RMSE `2e-5`

A Release CPU run of this test, gcc 13, 2 threads, native CPU instructions off,
was bitwise identical (`max_abs=0`, `rmse=0`, no 8-bit mismatches) through 512,
including the shapes below. That is a CPU result. It is not a Metal result.
`--bench 384` and `--bench 512` print milliseconds for the process that ran
them. Do not read them as M5 Ultra frame time.

Warm device calls from that CPU run, 33 input frames and a motion tail of 9:

| Size | Graph | Fused CPU |
|---|---|---|
| 384 | 2202 ms | 403 ms |
| 512 | 3915 ms | 664 ms |

The fused CPU kernel is faster than the CPU graph because it removes the long
chain of elementwise launches. It is slower than the Metal graph times already
measured on the M5 Ultra (about 196 ms at 384 and 349 ms at 512 for color
alone). On Metal the fused path dispatches the GPU shader; it does not run this
CPU kernel. That shader has not been timed here.

### Isolated Mac build and measurement

Build only this family so the binary stays local to the benchmark. Metal is on
by default in `scripts/build_metal.sh`:

```bash
scripts/build_metal.sh --deployment-build --model-set custom --models soulx_flashhead \
  --target soulx_flashhead_color_opt_test
./build/macos-metal-release/bin/soulx_flashhead_color_opt_test --backend metal --bench 384 --log /tmp/flashhead-color-384.log
./build/macos-metal-release/bin/soulx_flashhead_color_opt_test --backend metal --bench 512 --log /tmp/flashhead-color-512.log
```

`--backend metal` compares the Metal graph with the fused Metal kernel on the
same synthetic frames (33 input frames, motion tail 9, so 24 delivered frames
at 384 or 512). The process prints max absolute error, RMSE, 8-bit mismatch
count, and warm graph versus fused milliseconds. It exits non-zero if the error
exceeds max abs `1e-4` and RMSE `2e-5`. That check has not been run. A miss is
a reason to leave `color.impl` at `graph`. Passing it would still not be a
speedup measurement. MSL `log`, `exp`, and `sqrt` can differ from the CPU replica
even when the reduction matches.

A full Lite chunk still has to be timed with the GGUF, because this test does
not run the VAE or DiT:

```bash
./build/macos-metal-release/bin/audiocpp_cli \
  --family soulx_flashhead --task gen --backend metal --threads 2 \
  --model /path/to/SoulX-FlashHead-Lite-Q8_0.gguf \
  --audio /path/to/speech.wav \
  --request-option reference_image_file=/path/to/portrait.png \
  --request-option height=512 --request-option width=512 \
  --request-option num_inference_steps=2 \
  --session-option 'soulx_flashhead.exec_plan={"color":{"impl":"fused"}}' \
  --out-dir /tmp/flashhead-fused --log --log-file /tmp/flashhead-fused.log
```

Read `soulx_flashhead.color.wall_ms` and `ltx_video_vae.decode.graph.compute_ms`
from that log. Compare a second run that omits the session option. Rollback is
that second run. Nothing in this tree records an M5 Ultra delta: the cloud
build host has no Metal device, so GPU milliseconds, 25 FPS, and the 120-second
contention profile are unverified until those commands are run on the machine.
