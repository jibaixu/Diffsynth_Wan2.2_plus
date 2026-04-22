#!/usr/bin/env python
"""Export side-by-side track-map debug videos from the WAN training dataset."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffsynth.diffusion.parsers import (
    add_dataset_base_config,
    add_track_context_config,
    add_video_size_config,
)
from diffsynth.pipelines.wan_video_data import build_wan_video_dataset
from diffsynth.pipelines.wan_video_spec import WanRuntimeConfig


"""
examples/wanvideo/model_training/visualize_track_map_debug.py \
    --dataset_base_path /data_jbx/Codes/Diffsynth_Wan2.2_plus/data/4_4_four_tasks_wan \
    --dataset_metadata_path /data_jbx/Codes/Diffsynth_Wan2.2_plus/data/4_4_four_tasks_wan/meta/episodes_train.track_bert.jsonl \
    --output_dir ./track_map_vis \
    --max_samples 4
"""

DEFAULT_METADATA_CANDIDATES = (
    "meta/episodes_train.track_bert.jsonl",
    "meta/episodes_train.jsonl",
    "meta/episodes_train.json",
    "meta/episodes.jsonl",
)
DEFAULT_DATA_FILE_KEYS = ("video", "track")
DEFAULT_OUTPUT_DIR = ROOT_DIR / "debug" / "track_map_vis"
DEFAULT_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize original frames, rendered track maps, and overlays from the WAN dataset.",
    )
    parser = add_dataset_base_config(parser)
    parser = add_video_size_config(parser)
    parser = add_track_context_config(parser)
    parser.set_defaults(
        height=480,
        width=640,
        spatial_division_factor=32,
        num_frames=17,
        num_history_frames=1,
        history_template_sampling=0,
        track_apply_noise=1,
        track_noise_corrupt_ratio=0.3,
        track_noise_offset_scale=0.008,
        track_noise_drift_scale=0.002,
        track_noise_dropout_ratio=0.1,
        track_noise_warmup_frames=3,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for exported MP4 previews.",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Comma-separated sample indices or inclusive ranges, e.g. 0,3,8-10.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=4,
        help="Number of leading metadata rows to export when --sample_indices is not set. <=0 means all rows.",
    )
    parser.add_argument(
        "--view_index",
        type=int,
        default=0,
        help="Which camera view to export from the multi-view tensors.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.35,
        help="Track-map weight inside the overlay panel.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=5,
        help="ImageIO quality passed to the MP4 writer.",
    )
    parser.add_argument(
        "--skip_existing",
        type=int,
        choices=[0, 1],
        default=1,
        help="Skip outputs that already exist.",
    )
    return parser


def resolve_metadata_path(dataset_base_path: str, dataset_metadata_path: str | None) -> Path:
    if dataset_metadata_path:
        path = Path(dataset_metadata_path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return path

    base_path = Path(dataset_base_path)
    for relative_path in DEFAULT_METADATA_CANDIDATES:
        candidate = base_path / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not infer dataset metadata path. Set --dataset_metadata_path explicitly."
    )


def read_metadata_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected list metadata in {path}, got {type(payload).__name__}")
        return payload
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported metadata format: {path}")


def parse_sample_indices(spec: str | None, total_size: int) -> list[int] | None:
    if spec is None or spec.strip() == "":
        return None

    selected: list[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid sample range: {item}")
            values = range(start, end + 1)
        else:
            values = (int(item),)
        for value in values:
            if value < 0 or value >= total_size:
                raise IndexError(f"sample index {value} out of range for metadata size {total_size}")
            if value not in seen:
                selected.append(value)
                seen.add(value)
    if len(selected) == 0:
        raise ValueError("No valid sample indices were parsed.")
    return selected


def select_sample_indices(args, total_size: int) -> list[int]:
    selected = parse_sample_indices(args.sample_indices, total_size)
    if selected is not None:
        return selected
    if total_size <= 0:
        return []
    if int(args.max_samples) <= 0:
        return list(range(total_size))
    return list(range(min(int(args.max_samples), total_size)))


def build_track_runtime() -> WanRuntimeConfig:
    return WanRuntimeConfig(
        modules=("trackctx",),
        module_bases=("trackctx",),
        text_mode="off",
        action_mode="off",
        image_mode="off",
        track_context_enabled=True,
        enable_text=False,
        enable_text_encoder=False,
        action_enabled=False,
        has_text_input_for_dit=False,
        clip_mode=0,
        data_file_keys=DEFAULT_DATA_FILE_KEYS,
        model_paths=tuple(),
        tokenizer_path=None,
    )


def build_dataset(args, metadata_path: Path, sample_indices: Sequence[int]):
    runtime = build_track_runtime()
    return build_wan_video_dataset(
        runtime,
        base_path=args.dataset_base_path,
        metadata_path=str(metadata_path),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        num_history_frames=int(args.num_history_frames),
        repeat=1,
        resize_mode=args.resize_mode,
        max_pixels=int(args.max_pixels),
        data_file_keys=DEFAULT_DATA_FILE_KEYS,
        sample_indices=list(sample_indices),
        history_template_sampling=bool(args.history_template_sampling),
        height_division_factor=int(args.spatial_division_factor),
        width_division_factor=int(args.spatial_division_factor),
        time_division_factor=4,
        time_division_remainder=1,
        track_num_points=int(args.track_num_points),
        track_point_radius=int(args.track_point_radius),
        track_seed=int(args.track_seed),
        track_apply_noise=bool(args.track_apply_noise),
        track_noise_corrupt_ratio=float(args.track_noise_corrupt_ratio),
        track_noise_offset_scale=float(args.track_noise_offset_scale),
        track_noise_drift_scale=float(args.track_noise_drift_scale),
        track_noise_dropout_ratio=float(args.track_noise_dropout_ratio),
        track_noise_warmup_frames=int(args.track_noise_warmup_frames),
    )


def frame_tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    if not isinstance(frame, torch.Tensor):
        frame = torch.as_tensor(frame)
    array = frame.detach().cpu().float().clamp(-1.0, 1.0).permute(1, 2, 0).numpy()
    array = np.clip(np.rint((array + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return array


def build_overlay_frame(image: np.ndarray, track_map: np.ndarray, alpha: float) -> np.ndarray:
    if image.shape != track_map.shape:
        raise ValueError(f"Mismatched frame shapes: {image.shape} vs {track_map.shape}")
    mask = np.any(track_map > 0, axis=2, keepdims=True)
    overlay = image.astype(np.float32)
    if np.any(mask):
        blended = (1.0 - alpha) * image.astype(np.float32) + alpha * track_map.astype(np.float32)
        overlay = np.where(mask, blended, overlay)
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def compose_triptych_frame(image: np.ndarray, track_map: np.ndarray, alpha: float) -> np.ndarray:
    overlay = build_overlay_frame(image, track_map, alpha=alpha)
    return np.concatenate((image, track_map, overlay), axis=1)


def format_metadata_int(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    try:
        return f"{int(value):06d}"
    except (TypeError, ValueError):
        return fallback


def build_output_path(output_dir: Path, sample_index: int, row: dict[str, Any], view_index: int) -> Path:
    episode_id = format_metadata_int(row.get("episode_index"), "unknown")
    start_frame = format_metadata_int(row.get("start_frame"), "start")
    end_frame = format_metadata_int(row.get("end_frame"), "end")
    name = f"sample_{sample_index:06d}_ep_{episode_id}_start_{start_frame}_end_{end_frame}_view{view_index}.mp4"
    return output_dir / name


def export_sample_video(
    sample: dict[str, Any],
    output_path: Path,
    *,
    view_index: int,
    fps: int,
    quality: int,
    overlay_alpha: float,
) -> None:
    if "video" not in sample or "track" not in sample:
        raise KeyError("Sample must contain both `video` and `track` tensors.")

    video = torch.as_tensor(sample["video"])
    track = torch.as_tensor(sample["track"])
    if video.ndim != 5 or track.ndim != 5:
        raise ValueError(
            f"Expected video/track tensors with shape (V,C,T,H,W), got {tuple(video.shape)} and {tuple(track.shape)}"
        )
    if int(view_index) < 0 or int(view_index) >= int(video.shape[0]):
        raise IndexError(f"view_index={view_index} out of range for video views {int(video.shape[0])}")
    if int(view_index) >= int(track.shape[0]):
        raise IndexError(f"view_index={view_index} out of range for track views {int(track.shape[0])}")
    if video.shape[2] != track.shape[2]:
        raise ValueError(
            f"Video/track frame count mismatch: video T={int(video.shape[2])}, track T={int(track.shape[2])}"
        )

    frames: list[np.ndarray] = []
    for frame_id in range(int(video.shape[2])):
        image = frame_tensor_to_uint8(video[view_index, :, frame_id])
        track_map = frame_tensor_to_uint8(track[view_index, :, frame_id])
        frames.append(compose_triptych_frame(image, track_map, alpha=overlay_alpha))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        writer = imageio.get_writer(
            str(output_path),
            fps=int(fps),
            quality=int(quality),
            codec=DEFAULT_CODEC,
            pixelformat=DEFAULT_PIXEL_FORMAT,
            ffmpeg_log_level="error",
        )
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to export {output_path}. Ensure ffmpeg is available with libx264 support."
        ) from exc
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not 0.0 <= float(args.overlay_alpha) <= 1.0:
        raise ValueError("--overlay_alpha must be in [0, 1].")

    metadata_path = resolve_metadata_path(args.dataset_base_path, args.dataset_metadata_path)
    rows = read_metadata_rows(metadata_path)
    sample_indices = select_sample_indices(args, len(rows))
    if len(sample_indices) == 0:
        raise ValueError("No samples selected for export.")

    dataset = build_dataset(args, metadata_path, sample_indices)
    output_dir = Path(args.output_dir)

    torch.set_grad_enabled(False)
    saved_count = 0
    skipped_count = 0
    for local_index, sample_index in enumerate(tqdm(sample_indices, desc="Exporting samples")):
        output_path = build_output_path(output_dir, sample_index, rows[sample_index], int(args.view_index))
        if bool(args.skip_existing) and output_path.is_file():
            skipped_count += 1
            continue
        sample = dataset[local_index]
        export_sample_video(
            sample,
            output_path,
            view_index=int(args.view_index),
            fps=int(args.fps),
            quality=int(args.quality),
            overlay_alpha=float(args.overlay_alpha),
        )
        saved_count += 1

    print(f"Metadata path: {metadata_path}")
    print(f"Selected samples: {len(sample_indices)}")
    print(f"Saved videos: {saved_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
