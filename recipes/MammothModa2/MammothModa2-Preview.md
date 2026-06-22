# MammothModa2-Preview

> MammothModa2-Preview text-to-image generation through the shared offline image example

## Summary

- Vendor: ByteDance Research
- Model: `bytedance-research/MammothModa2-Preview`
- Task: Text-to-image generation (AR → DiT two-stage pipeline)
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to run MammothModa2-Preview text-to-image through the shared
offline image example (`text_to_image.py`) instead of a model-specific script.
The generic example formats the AR prompt, drives the AR → DiT stage pipeline,
and forwards MammothModa2-specific generation parameters through the
pipeline-declared `extra_body` contract.

MammothModa2's DiT stage consumes its inputs through the multi-stage kwargs
interface (not `OmniDiffusionRequest`), so its generation knobs
(`text_guidance_scale`, `cfg_range`, `num_inference_steps`) are passed via
`--extra-body` rather than the standard `--num-inference-steps` / `--cfg-scale`
flags. Image size uses the standard `--height` / `--width` flags.

## References

- Upstream model:
  [`bytedance-research/MammothModa2-Preview`](https://huggingface.co/bytedance-research/MammothModa2-Preview)
- Related offline example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Declared parameters:
  [`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)
- Stage config:
  [`vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml`](../../vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml)

## Hardware Support

The default stage config runs both the AR and DiT stages on a single GPU
(`devices: "0"`). A single GPU with ≥40 GB of VRAM is sufficient.

## GPU

### 1x L40S 48GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with one ≥40 GB GPU (e.g. L40S 48 GB)
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Offline Commands

Download the model:

```bash
hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

Run text-to-image with the shared offline example from the repository root. The
stage config sets `trust_remote_code`, so no extra flag is needed. Forward the
MammothModa2 generation parameters as a JSON object through `--extra-body`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
  --output mammoth_t2i.png
```

The `--extra-body` JSON forwards MammothModa2-specific parameters into
`OmniDiffusionSamplingParams.extra_args`. Keys are filtered against the model's
declared `extra_body_params` (see
[`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)),
so unknown keys for MammothModa2 are silently dropped:

- `text_guidance_scale` — classifier-free guidance scale for the DiT stage
  (default `9.0`; CFG is active only when `> 1.0`).
- `cfg_range` — relative step range `[start, end]` over which CFG is applied
  (default `[0.0, 1.0]`).
- `num_inference_steps` — number of DiT denoising steps (default `50`).

`--height` and `--width` must be multiples of 16.

#### Verification

The example writes the generated image to the `--output` path. Confirm the file
exists and is a valid image:

```bash
ls -lh mammoth_t2i.png
python -c "from PIL import Image; print(Image.open('mammoth_t2i.png').size)"
```
