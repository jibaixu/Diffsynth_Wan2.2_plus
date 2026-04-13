#!/usr/bin/env python
"""
Single-episode autoregressive WAN inference with training-equivalent history noise.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from diffsynth.diffusion.parsers import (
    add_action_config,
    add_dataset_base_config,
    add_infer_config,
    add_model_config,
    add_training_config,
    add_video_size_config,
    build_grouped_config,
)
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.pipelines.wan_video_data import (
    WAN_INFERENCE_DATASET_NUM_FRAMES,
    build_wan_video_dataset,
)
from diffsynth.utils.data import save_video

try:
    from inference_support import (
        CheckpointPipelineManager,
        FrameConverter,
        VideoSaver,
        build_wan_inference_config,
        load_flat_config_defaults,
        resolve_optional_path,
    )
except ModuleNotFoundError:
    from .inference_support import (
        CheckpointPipelineManager,
        FrameConverter,
        VideoSaver,
        build_wan_inference_config,
        load_flat_config_defaults,
        resolve_optional_path,
    )


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("infer_single_episode_autoreg")


def parse_args():
    parser = argparse.ArgumentParser(description="Single-episode autoregressive WAN inference")
    parser = add_model_config(parser)
    parser = add_dataset_base_config(parser)
    parser = add_action_config(parser)
    parser = add_video_size_config(parser)
    parser = add_training_config(parser)
    parser = add_infer_config(parser)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for this episode.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device used for inference.")
    parser.add_argument(
        "--show_progress",
        type=int,
        choices=[0, 1],
        default=1,
        help="Show per-chunk denoising and save progress bars.",
    )
    selector = parser.add_mutually_exclusive_group(required=False)
    selector.add_argument("--episode_index", type=int, default=None, help="Metadata episode_index to roll out.")
    selector.add_argument("--sample_index", type=int, default=None, help="Zero-based metadata row index to roll out.")

    for action in parser._actions:
        if getattr(action, "dest", None) == "dataset_base_path":
            action.required = False
            break

    pre_args, _ = parser.parse_known_args()
    defaults = load_flat_config_defaults(pre_args.checkpoint_path)
    if defaults:
        known = {action.dest for action in parser._actions if getattr(action, "dest", None)}
        parser.set_defaults(**{key: value for key, value in defaults.items() if key in known})

    args = parser.parse_args()
    if not args.dataset_base_path:
        raise ValueError("`--dataset_base_path` is required (or provide a config.json next to the checkpoint).")
    if args.episode_index is None and args.sample_index is None:
        raise ValueError("One of `--episode_index` or `--sample_index` is required.")

    grouped_config = build_grouped_config(parser, args) or {}
    return build_wan_inference_config(vars(args).copy(), grouped_config=grouped_config)


def read_metadata_rows(path: str) -> List[Dict[str, Any]]:
    metadata_path = Path(path)
    suffix = metadata_path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with metadata_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list metadata in {path}, got {type(payload).__name__}")
        return payload
    if suffix == ".csv":
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported metadata format: {path}")


def select_metadata_row(
    rows: Sequence[Dict[str, Any]],
    *,
    episode_index: Optional[int],
    sample_index: Optional[int],
) -> Tuple[int, Dict[str, Any]]:
    if sample_index is not None:
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= len(rows):
            raise IndexError(f"sample_index={sample_index} out of range for metadata size {len(rows)}")
        return sample_index, rows[sample_index]

    target = int(episode_index)
    matches = [
        (index, row)
        for index, row in enumerate(rows)
        if row.get("episode_index") is not None and int(row["episode_index"]) == target
    ]
    if len(matches) == 0:
        raise KeyError(f"episode_index={target} not found in metadata")
    if len(matches) > 1:
        raise ValueError(f"episode_index={target} matched multiple rows; use --sample_index instead.")
    return matches[0]


def build_full_episode_dataset(config, sample_index: int):
    spatial_division_factor = int(getattr(config, "spatial_division_factor", 16))
    return build_wan_video_dataset(
        config.runtime,
        base_path=config.dataset_base_path,
        metadata_path=config.dataset_metadata_path,
        height=config.height,
        width=config.width,
        num_frames=int(config.num_frames),
        num_history_frames=int(config.num_history_frames),
        repeat=1,
        resize_mode=config.resize_mode,
        max_pixels=config.max_pixels,
        data_file_keys=config.data_file_keys,
        dataset_num_frames=WAN_INFERENCE_DATASET_NUM_FRAMES,
        sample_indices=[sample_index],
        action_stat_path=config.action_stat_path,
        action_type=config.action_type,
        history_template_sampling=0,
        height_division_factor=spatial_division_factor,
        width_division_factor=spatial_division_factor,
        time_division_factor=4,
        time_division_remainder=1,
    )


def align_num_frames(num_frames: int) -> int:
    if (num_frames - 1) % 4 == 0:
        return num_frames
    return num_frames + (4 - ((num_frames - 1) % 4))


def build_history_indices(
    num_generated_frames: int,
    num_history_frames: int,
    use_history_template: bool,
) -> List[int]:
    if num_history_frames <= 0:
        return []
    if num_generated_frames < num_history_frames:
        raise ValueError(
            f"Need at least {num_history_frames} generated frames, got {num_generated_frames}"
        )
    if use_history_template and num_history_frames > 1:
        return [0] + list(range(num_generated_frames - (num_history_frames - 1), num_generated_frames))
    return list(range(num_generated_frames - num_history_frames, num_generated_frames))


def build_action_condition(
    action: np.ndarray,
    *,
    history_indices: Sequence[int],
    future_start: int,
    future_count: int,
    infer_frames: int,
) -> Tuple[np.ndarray, List[int], int]:
    history_action = action[:, history_indices, :]
    future_indices = list(range(future_start, future_start + future_count))
    future_action = action[:, future_start : future_start + future_count, :]
    action_cond = np.concatenate([history_action, future_action], axis=1)
    pad_frames = int(infer_frames - action_cond.shape[1])
    if pad_frames > 0:
        pad = np.repeat(action_cond[:, -1:, :], repeats=pad_frames, axis=1)
        action_cond = np.concatenate([action_cond, pad], axis=1)
    return action_cond.astype(np.float32), future_indices, pad_frames


def sanitize_json_value(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def resolve_prompt_inputs(sample: Dict[str, Any], config) -> Tuple[str, Optional[str], str, Optional[str]]:
    prompt = str(sample.get("prompt", ""))
    prompt_emb = sample.get("prompt_emb")
    if prompt_emb is not None:
        prompt_emb = resolve_optional_path(prompt_emb, config.dataset_base_path)
    negative_prompt = str(getattr(config, "negative_prompt", "") or "")
    negative_prompt_emb = resolve_optional_path(config.negative_prompt_emb, config.dataset_base_path)

    if config.runtime.text_mode == "emb":
        if not prompt_emb or not os.path.isfile(prompt_emb):
            raise FileNotFoundError(f"Missing prompt_emb for episode {sample.get('episode_index')}: {prompt_emb}")
        if not negative_prompt_emb or not os.path.isfile(negative_prompt_emb):
            raise FileNotFoundError(f"Missing negative_prompt_emb: {negative_prompt_emb}")
    elif not config.runtime.enable_text:
        prompt = ""
        prompt_emb = None
        negative_prompt = ""
        negative_prompt_emb = None

    return prompt, prompt_emb, negative_prompt, negative_prompt_emb


def save_stacked_video(
    video: torch.Tensor,
    output_path: Path,
    *,
    fps: int,
    quality: int,
    show_progress: bool,
) -> Path:
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise TypeError("`video` must be a torch.Tensor with shape (V,C,T,H,W).")

    converter = FrameConverter()
    frames: List[np.ndarray] = []
    video = video.detach().to(dtype=torch.float32).cpu()
    for frame_idx in range(int(video.shape[2])):
        rows = []
        for view_idx in range(int(video.shape[0])):
            frame = converter.to_uint8(video[view_idx, :, frame_idx])
            rows.append(converter.ensure_rgb(frame))
        frames.append(np.vstack(rows))

    save_video(
        np.asarray(frames),
        str(output_path),
        fps=fps,
        quality=quality,
        show_progress=show_progress,
    )
    return output_path


@torch.no_grad()
def generate_chunk_with_history_noise(
    pipeline: WanVideoPipeline,
    *,
    prompt: str,
    prompt_emb: Optional[str],
    negative_prompt: str,
    negative_prompt_emb: Optional[str],
    input_video: torch.Tensor,
    action_cond: np.ndarray,
    height: int,
    width: int,
    infer_frames: int,
    num_history_frames: int,
    cfg_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
    show_progress: bool,
    sigma_shift: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not isinstance(input_video, torch.Tensor) or input_video.ndim != 5:
        raise TypeError("`input_video` must be a torch.Tensor with shape (V,C,T,H,W).")

    input_video_cpu = input_video.detach().to(dtype=torch.float32).cpu()
    pipeline.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)

    inputs_posi = {
        "prompt": prompt,
        "prompt_emb": prompt_emb,
        "num_inference_steps": num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "prompt_emb": negative_prompt_emb,
        "num_inference_steps": num_inference_steps,
    }
    inputs_shared = {
        "input_video": input_video,
        "denoising_strength": 1.0,
        "num_views": int(input_video.shape[0]),
        "seed": seed,
        "rand_device": "cpu",
        "height": int(height),
        "width": int(width),
        "num_frames": int(infer_frames),
        "num_history_frames": int(num_history_frames),
        "action": action_cond,
        "cfg_scale": float(cfg_scale),
        "sigma_shift": float(sigma_shift),
        "tiled": False,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
    }
    for unit in pipeline.units:
        inputs_shared, inputs_posi, inputs_nega = pipeline.unit_runner(
            unit,
            pipeline,
            inputs_shared,
            inputs_posi,
            inputs_nega,
        )

    base_history_latents = inputs_shared.get("first_frame_latents")
    if not isinstance(base_history_latents, torch.Tensor):
        raise RuntimeError("Expected `first_frame_latents` from fused TI2V conditioning, but it was missing.")
    if base_history_latents.dim() == 4:
        base_history_latents = base_history_latents.unsqueeze(0)
    base_history_latents = base_history_latents.to(dtype=pipeline.torch_dtype, device=pipeline.device)

    latents = inputs_shared["latents"]
    noise = inputs_shared["noise"]
    history_t = min(int(base_history_latents.shape[2]), int(latents.shape[2]))
    conditioning_latents = base_history_latents[:, :, :history_t].clone()
    small_timestep_idx = None

    if history_t > 0:
        latents[:, :, :history_t] = conditioning_latents
    if pipeline.action_injection_mode == "adaln" and history_t > 1:
        small_timestep_idx = max(0, len(pipeline.scheduler.timesteps) - 50)
        small_timestep = pipeline.scheduler.timesteps[small_timestep_idx].unsqueeze(0).to(
            dtype=pipeline.torch_dtype,
            device=pipeline.device,
        )
        conditioning_latents[:, :, 1:history_t] = pipeline.scheduler.add_noise(
            conditioning_latents[:, :, 1:history_t],
            noise[:, :, 1:history_t],
            small_timestep,
        )
        latents[:, :, 1:history_t] = conditioning_latents[:, :, 1:history_t]
    inputs_shared["latents"] = latents

    pipeline.load_models_to_device(pipeline.in_iteration_models)
    models = {name: getattr(pipeline, name) for name in pipeline.in_iteration_models}
    iterator = tqdm(
        pipeline.scheduler.timesteps,
        desc="Denoising",
        disable=not show_progress,
    )
    for progress_id, timestep in enumerate(iterator):
        timestep = timestep.unsqueeze(0).to(dtype=pipeline.torch_dtype, device=pipeline.device)
        noise_pred_posi = pipeline.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        if cfg_scale != 1.0:
            noise_pred_nega = pipeline.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        inputs_shared["latents"] = pipeline.scheduler.step(
            noise_pred,
            pipeline.scheduler.timesteps[progress_id],
            inputs_shared["latents"],
        )
        if history_t > 0:
            inputs_shared["latents"][:, :, :history_t] = conditioning_latents[:, :, :history_t]

    pipeline.load_models_to_device(["vae"])
    latents = inputs_shared["latents"]
    num_views = int(inputs_shared.get("num_views", 1))
    if latents.shape[-2] % num_views != 0:
        raise ValueError(f"Latent height {latents.shape[-2]} is not divisible by num_views={num_views}.")
    latents_by_view = rearrange(
        latents,
        "b c t (v h) w -> (b v) c t h w",
        v=num_views,
        h=latents.shape[-2] // num_views,
    )
    predicted_video = pipeline.vae.decode(
        latents_by_view,
        device=pipeline.device,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    pipeline.load_models_to_device([])

    predicted_video = predicted_video.detach().to(dtype=torch.float32).cpu()
    history_to_copy = min(
        int(num_history_frames),
        int(predicted_video.shape[2]),
        int(input_video_cpu.shape[2]),
    )
    if history_to_copy > 0:
        predicted_video[:, :, :history_to_copy] = input_video_cpu[:, :, :history_to_copy]

    return predicted_video, {
        "history_t": int(history_t),
        "small_timestep_idx": small_timestep_idx,
        "num_scheduler_steps": len(pipeline.scheduler.timesteps),
    }


def rollout_episode(
    pipeline: WanVideoPipeline,
    config,
    sample: Dict[str, Any],
    *,
    show_progress: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    gt_video = sample["video"]
    if not isinstance(gt_video, torch.Tensor) or gt_video.ndim != 5:
        raise TypeError("Expected `sample['video']` to be a (V,C,T,H,W) tensor.")
    gt_video = gt_video.detach().to(dtype=torch.float32).cpu()

    action = sample.get("action")
    if action is None:
        raise ValueError("Autoregressive rollout requires action conditioning, but `sample['action']` is missing.")
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 3 or action.shape[0] != 1:
        raise ValueError(f"Expected action shape (1,T,14), got {action.shape}")

    total_frames = min(int(gt_video.shape[2]), int(action.shape[1]))
    gt_video = gt_video[:, :, :total_frames]
    action = action[:, :total_frames, :]

    num_history_frames = int(config.num_history_frames)
    num_frames = int(config.num_frames)
    future_frames = int(num_frames - num_history_frames)
    if future_frames <= 0:
        raise ValueError(
            f"Invalid frame setup: num_frames={num_frames}, num_history_frames={num_history_frames}"
        )
    if total_frames < num_history_frames:
        raise ValueError(
            f"Episode has {total_frames} aligned frames, smaller than num_history_frames={num_history_frames}"
        )

    prompt, prompt_emb, negative_prompt, negative_prompt_emb = resolve_prompt_inputs(sample, config)
    use_history_template = bool(int(getattr(config, "history_template_sampling", 0)))

    generated_frames: List[torch.Tensor] = []
    for frame_idx in range(num_history_frames):
        generated_frames.append(gt_video[:, :, frame_idx].clone())

    chunk_traces: List[Dict[str, Any]] = []
    while len(generated_frames) < total_frames:
        chunk_idx = len(chunk_traces)
        future_start = len(generated_frames)
        remaining_future = total_frames - future_start
        current_future = min(future_frames, remaining_future)
        requested_frames = num_history_frames + current_future
        infer_frames = align_num_frames(requested_frames)

        history_indices = build_history_indices(
            len(generated_frames),
            num_history_frames,
            use_history_template,
        )
        history_frame_seq = [generated_frames[index] for index in history_indices]
        input_video = torch.stack(history_frame_seq, dim=2)

        action_cond, future_indices, action_pad_frames = build_action_condition(
            action,
            history_indices=history_indices,
            future_start=future_start,
            future_count=current_future,
            infer_frames=infer_frames,
        )

        chunk_seed = None if config.seed is None else int(config.seed) + chunk_idx
        predicted_chunk, chunk_debug = generate_chunk_with_history_noise(
            pipeline,
            prompt=prompt,
            prompt_emb=prompt_emb,
            negative_prompt=negative_prompt,
            negative_prompt_emb=negative_prompt_emb,
            input_video=input_video,
            action_cond=action_cond,
            height=int(gt_video.shape[-2]),
            width=int(gt_video.shape[-1]),
            infer_frames=infer_frames,
            num_history_frames=num_history_frames,
            cfg_scale=float(config.cfg_scale),
            num_inference_steps=int(config.num_inference_steps),
            seed=chunk_seed,
            show_progress=show_progress,
        )

        future_video = predicted_chunk[:, :, num_history_frames:]
        append_count = min(current_future, int(future_video.shape[2]))
        for frame_idx in range(append_count):
            generated_frames.append(future_video[:, :, frame_idx].clone())

        chunk_traces.append(
            {
                "chunk_index": int(chunk_idx),
                "chunk_seed": chunk_seed,
                "generated_frames_before_chunk": int(future_start),
                "history_indices": [int(index) for index in history_indices],
                "future_indices": [int(index) for index in future_indices],
                "requested_frames": int(requested_frames),
                "infer_frames": int(infer_frames),
                "action_pad_frames": int(action_pad_frames),
                "future_frames_requested": int(current_future),
                "future_frames_appended": int(append_count),
                "generated_frames_after_chunk": int(len(generated_frames)),
                "history_latent_frames": int(chunk_debug["history_t"]),
                "small_timestep_idx": chunk_debug["small_timestep_idx"],
                "num_scheduler_steps": int(chunk_debug["num_scheduler_steps"]),
            }
        )

    predicted_video = torch.stack(generated_frames, dim=2)[:, :, :total_frames]
    return predicted_video, {
        "total_frames": int(total_frames),
        "num_history_frames": int(num_history_frames),
        "num_frames": int(num_frames),
        "future_frames": int(future_frames),
        "use_history_template": bool(use_history_template),
        "chunks": chunk_traces,
    }


def resolve_output_dir(config, sample_index: int, episode_index: int) -> Path:
    explicit_output = config.values.get("output_dir")
    if explicit_output:
        return Path(explicit_output)
    checkpoint_path = config.values.get("checkpoint_path")
    if checkpoint_path:
        base_dir = Path(checkpoint_path).parent
    else:
        base_dir = Path("Ckpt") / "pretrained"
    return base_dir / "single_episode_autoreg" / f"sample_{sample_index:06d}_ep{episode_index}"


def main() -> None:
    config = parse_args()
    logger = setup_logger()

    metadata_rows = read_metadata_rows(config.dataset_metadata_path)
    selected_sample_index, selected_row = select_metadata_row(
        metadata_rows,
        episode_index=config.values.get("episode_index"),
        sample_index=config.values.get("sample_index"),
    )
    selected_episode_index = int(selected_row["episode_index"])
    logger.info(
        "Selected metadata row: sample_index=%s, episode_index=%s",
        selected_sample_index,
        selected_episode_index,
    )

    dataset = build_full_episode_dataset(config, selected_sample_index)
    sample = dataset[0]
    logger.info("Loaded full episode sample with video shape: %s", tuple(sample["video"].shape))
    if sample.get("action") is not None:
        logger.info("Loaded action shape: %s", tuple(np.asarray(sample["action"]).shape))

    manager = CheckpointPipelineManager(
        config,
        logger,
        device="cpu" if getattr(config, "initialize_model_on_cpu", False) else config.values.get("device", "cuda"),
        verbose=True,
    )
    checkpoints = manager.discover_checkpoints()
    pipeline = manager.initialize_pipeline(checkpoints)
    manager.update_checkpoint(checkpoints[0] if checkpoints else None)

    predicted_video, rollout_trace = rollout_episode(
        pipeline,
        config,
        sample,
        show_progress=bool(int(config.values.get("show_progress", 1))),
    )
    gt_video = sample["video"].detach().to(dtype=torch.float32).cpu()[:, :, : predicted_video.shape[2]]

    output_dir = resolve_output_dir(config, selected_sample_index, selected_episode_index)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = save_stacked_video(
        predicted_video,
        output_dir / "pred.mp4",
        fps=int(config.fps),
        quality=int(config.quality),
        show_progress=bool(int(config.values.get("show_progress", 1))),
    )
    compare_path = VideoSaver(
        fps=int(config.fps),
        quality=int(config.quality),
        show_progress=bool(int(config.values.get("show_progress", 1))),
    ).save_comparison(
        gt_video,
        predicted_video,
        output_dir,
        "compare.mp4",
    )

    trace_payload = {
        "selected_sample_index": int(selected_sample_index),
        "selected_episode_index": int(selected_episode_index),
        "metadata_path": str(Path(config.dataset_metadata_path).resolve()),
        "selected_metadata_row": sanitize_json_value(selected_row),
        "raw_length": int(selected_row.get("raw_length", selected_row.get("length", predicted_video.shape[2]))),
        "aligned_total_frames": int(predicted_video.shape[2]),
        "effective_config": sanitize_json_value(config.grouped_config or config.values),
        "outputs": {
            "pred_mp4": str(pred_path.resolve()),
            "compare_mp4": str(compare_path.resolve()),
        },
        "rollout": sanitize_json_value(rollout_trace),
    }
    trace_path = output_dir / "trace.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(trace_payload, f, indent=2, ensure_ascii=False)

    logger.info("Saved pred video: %s", pred_path.resolve())
    logger.info("Saved compare video: %s", compare_path.resolve())
    logger.info("Saved trace: %s", trace_path.resolve())


if __name__ == "__main__":
    main()
