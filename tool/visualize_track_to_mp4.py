from __future__ import annotations

import argparse
import colorsys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


DEFAULT_META_PATH = "data/4_4_four_tasks_wan/meta/episodes_train.track_bert.jsonl"
DEFAULT_OUTPUT_PATH = "track_visualization_episode_000000.mp4"
DEFAULT_WAN_VAE_PATH = "/data1/modelscope/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
DEFAULT_NUM_POINTS = 256
DEFAULT_POINT_RADIUS = 6
DEFAULT_SEED = 42
DEFAULT_VIDEO_QUALITY = 8
DEFAULT_TILE_SIZE = (20, 20)
DEFAULT_TILE_STRIDE = (10, 10)
DEFAULT_FFMPEG_PARAMS = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]


@dataclass
class EpisodeRecord:
    record: dict[str, Any]
    dataset_root: Path


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class TrackView:
    tracks: np.ndarray
    vis: np.ndarray
    point_indices: np.ndarray
    colors_bgr: np.ndarray
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return int(self.tracks.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render sampled track points to a stacked MP4 and optionally encode it with Wan2.2 VAE."
    )
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH, help="Path to the dataset jsonl file.")
    parser.add_argument("--index", type=int, default=0, help="Episode index inside the jsonl file.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output mp4 path for the stacked track visualization video.",
    )
    parser.add_argument(
        "--input-video",
        default=None,
        help="Existing stacked mp4 path to encode. If unset, the script renders a new one first.",
    )
    parser.add_argument(
        "--latent-output",
        default=None,
        help="Output path for the encoded latent .pt file. Defaults to <video_stem>_latents.pt.",
    )
    parser.add_argument(
        "--vae-path",
        default=DEFAULT_WAN_VAE_PATH,
        help="Path to Wan2.2_VAE.pth.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help="Maximum number of visible points to sample per view.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=DEFAULT_POINT_RADIUS,
        help="Circle radius for each point. Radius 6 is the default for Wan2.2 VAE38 compatibility.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed used for per-view point sampling.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_VIDEO_QUALITY,
        help="ImageIO quality value for ffmpeg output.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for VAE encode. Use 'auto', 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Tensor dtype used for VAE encode. Auto picks float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--tile-size",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=DEFAULT_TILE_SIZE,
        help="Wan VAE tiled encode tile size in latent units.",
    )
    parser.add_argument(
        "--tile-stride",
        nargs=2,
        type=int,
        metavar=("H", "W"),
        default=DEFAULT_TILE_STRIDE,
        help="Wan VAE tiled encode tile stride in latent units.",
    )
    parser.add_argument(
        "--no-tiled",
        action="store_true",
        help="Disable tiled VAE encode.",
    )
    parser.add_argument(
        "--skip-latent",
        action="store_true",
        help="Only render the MP4 and skip Wan2.2 VAE encoding.",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="First frame index to encode from the stacked MP4.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to encode from the stacked MP4.",
    )
    return parser.parse_args()


def load_episode_record(meta_path: str | Path, index: int) -> EpisodeRecord:
    meta_path = Path(meta_path).resolve()
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")

    with meta_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if line_idx == index:
                record = json.loads(line)
                return EpisodeRecord(record=record, dataset_root=meta_path.parent.parent)

    raise IndexError(f"Could not find index {index} in {meta_path}")


def resolve_dataset_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (dataset_root / path)


def load_track_arrays(track_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    track_path = Path(track_path)
    with np.load(track_path) as data:
        tracks = np.asarray(data["tracks"], dtype=np.float32)
        vis = np.asarray(data["vis"], dtype=bool)
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Unexpected tracks shape {tracks.shape} in {track_path}")
    if vis.shape != tracks.shape[:2]:
        raise ValueError(f"vis shape {vis.shape} does not match tracks shape {tracks.shape} in {track_path}")
    return tracks, vis


def _count_frames_from_reader(reader, metadata: dict[str, Any]) -> int:
    nframes = metadata.get("nframes")
    if isinstance(nframes, (int, np.integer)) and int(nframes) > 0:
        return int(nframes)
    if isinstance(nframes, float) and np.isfinite(nframes) and nframes > 0:
        return int(round(nframes))
    try:
        frame_count = int(reader.count_frames())
        if frame_count > 0:
            return frame_count
    except Exception:
        pass

    duration = metadata.get("duration")
    fps = float(metadata.get("fps", 0.0))
    if duration is not None and fps > 0:
        frame_count = int(round(float(duration) * fps))
        if frame_count > 0:
            return frame_count
    raise RuntimeError("Failed to determine video frame count from metadata.")


def load_video_metadata(video_path: str | Path) -> VideoMetadata:
    reader = imageio.get_reader(str(video_path))
    try:
        metadata = reader.get_meta_data()
        width, height = metadata["size"]
        fps = float(metadata["fps"])
        frame_count = _count_frames_from_reader(reader, metadata)
    finally:
        reader.close()

    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"Invalid video metadata from {video_path}: "
            f"width={width}, height={height}, fps={fps}, frame_count={frame_count}"
        )
    return VideoMetadata(width=int(width), height=int(height), fps=fps, frame_count=frame_count)


def sample_visible_points(vis: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    visible_point_indices = np.flatnonzero(np.any(vis, axis=0))
    if visible_point_indices.size == 0:
        raise ValueError("No visible points were found in this view.")

    sample_size = min(num_points, visible_point_indices.size)
    sampled = rng.choice(visible_point_indices, size=sample_size, replace=False)
    return np.asarray(sampled, dtype=np.int64)


def generate_distinct_colors(num_colors: int) -> np.ndarray:
    if num_colors <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    colors: list[tuple[int, int, int]] = []
    used: set[tuple[int, int, int]] = set()
    golden_ratio = 0.6180339887498949
    idx = 0

    while len(colors) < num_colors:
        hue = (idx * golden_ratio) % 1.0
        saturation = 0.75 + 0.2 * ((idx % 3) / 2.0)
        value = 0.85 + 0.15 * (((idx // 3) % 3) / 2.0)
        rgb = tuple(int(round(channel * 255.0)) for channel in colorsys.hsv_to_rgb(hue, saturation, value))
        if rgb not in used and max(rgb) >= 96:
            used.add(rgb)
            colors.append((rgb[2], rgb[1], rgb[0]))
        idx += 1

    return np.asarray(colors, dtype=np.uint8)


def build_track_view(
    track_path: str | Path,
    metadata: VideoMetadata,
    num_points: int,
    seed: int,
) -> TrackView:
    tracks, vis = load_track_arrays(track_path)
    point_indices = sample_visible_points(vis, num_points, np.random.default_rng(seed))
    colors_bgr = generate_distinct_colors(len(point_indices))

    if tracks.shape[0] != metadata.frame_count:
        raise ValueError(
            f"Track frame count {tracks.shape[0]} does not match video frame count {metadata.frame_count} "
            f"for {track_path}"
        )

    return TrackView(
        tracks=tracks,
        vis=vis,
        point_indices=point_indices,
        colors_bgr=colors_bgr,
        width=metadata.width,
        height=metadata.height,
    )


def render_track_frame(view: TrackView, frame_idx: int, point_radius: int) -> np.ndarray:
    canvas = np.zeros((view.height, view.width, 3), dtype=np.uint8)
    frame_points = view.tracks[frame_idx, view.point_indices]
    frame_vis = view.vis[frame_idx, view.point_indices]
    valid = frame_vis & np.isfinite(frame_points).all(axis=1)
    if not np.any(valid):
        return canvas

    pixel_x = np.rint(frame_points[:, 0] * (view.width - 1)).astype(np.int32)
    pixel_y = np.rint(frame_points[:, 1] * (view.height - 1)).astype(np.int32)
    valid &= pixel_x >= 0
    valid &= pixel_x < view.width
    valid &= pixel_y >= 0
    valid &= pixel_y < view.height

    for x, y, color in zip(pixel_x[valid], pixel_y[valid], view.colors_bgr[valid]):
        cv2.circle(
            canvas,
            center=(int(x), int(y)),
            radius=point_radius,
            color=tuple(int(channel) for channel in color),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def iter_stacked_frames(top_view: TrackView, bottom_view: TrackView, point_radius: int) -> Iterator[np.ndarray]:
    if top_view.frame_count != bottom_view.frame_count:
        raise ValueError(
            f"View frame counts do not match: {top_view.frame_count} vs {bottom_view.frame_count}"
        )

    for frame_idx in range(top_view.frame_count):
        top_frame = render_track_frame(top_view, frame_idx, point_radius)
        bottom_frame = render_track_frame(bottom_view, frame_idx, point_radius)
        yield np.concatenate([top_frame, bottom_frame], axis=0)


def save_video(
    frames: Iterator[np.ndarray],
    save_path: str | Path,
    fps: float,
    quality: int,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(save_path),
        fps=fps,
        codec="libx264",
        quality=quality,
        ffmpeg_params=DEFAULT_FFMPEG_PARAMS,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def render_episode_tracks_to_mp4(
    meta_path: str | Path,
    episode_index: int,
    output_path: str | Path,
    num_points: int = DEFAULT_NUM_POINTS,
    point_radius: int = DEFAULT_POINT_RADIUS,
    seed: int = DEFAULT_SEED,
    quality: int = DEFAULT_VIDEO_QUALITY,
) -> dict[str, Any]:
    episode = load_episode_record(meta_path, episode_index)
    record = episode.record
    track_entries = record.get("track")
    video_entries = record.get("video")

    if not isinstance(track_entries, list) or len(track_entries) != 2:
        raise ValueError(f"Expected exactly 2 track paths, got {track_entries}")
    if not isinstance(video_entries, list) or len(video_entries) != 2:
        raise ValueError(f"Expected exactly 2 video paths, got {video_entries}")

    video_paths = [resolve_dataset_path(episode.dataset_root, path) for path in video_entries]
    track_paths = [resolve_dataset_path(episode.dataset_root, path) for path in track_entries]
    metadatas = [load_video_metadata(path) for path in video_paths]

    if metadatas[0] != metadatas[1]:
        raise ValueError(f"Two views have mismatched metadata: {metadatas[0]} vs {metadatas[1]}")

    top_view = build_track_view(track_paths[0], metadatas[0], num_points=num_points, seed=seed)
    bottom_view = build_track_view(track_paths[1], metadatas[1], num_points=num_points, seed=seed + 1)
    frames = iter_stacked_frames(top_view, bottom_view, point_radius=point_radius)
    save_video(frames, output_path, fps=metadatas[0].fps, quality=quality)

    return {
        "episode_index": episode_index,
        "prompt": record.get("prompt", ""),
        "output_path": str(Path(output_path).resolve()),
        "fps": metadatas[0].fps,
        "frame_count": metadatas[0].frame_count,
        "single_view_size": [metadatas[0].width, metadatas[0].height],
        "stacked_size": [metadatas[0].width, metadatas[0].height * 2],
        "num_points_per_view": [int(len(top_view.point_indices)), int(len(bottom_view.point_indices))],
        "track_paths": [str(path) for path in track_paths],
        "point_radius": int(point_radius),
    }


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[dtype_name]
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 VAE encode on CPU is not supported. Use float32 or bfloat16.")
    return dtype


def build_default_latent_output_path(video_path: str | Path) -> Path:
    video_path = Path(video_path)
    return video_path.with_name(f"{video_path.stem}_latents.pt")


def load_wan_ti2v_vae(
    vae_path: str | Path,
    device: str,
    dtype: torch.dtype,
):
    from diffsynth.models.wan_video_vae import WanVideoVAE38
    from diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter

    vae_path = Path(vae_path).resolve()
    state_dict = torch.load(str(vae_path), map_location="cpu")
    state_dict = WanVideoVAEStateDictConverter(state_dict)

    vae = WanVideoVAE38()
    missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=True)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Unexpected VAE state_dict mismatch. missing={missing_keys}, unexpected={unexpected_keys}"
        )

    vae = vae.to(device=device, dtype=dtype)
    vae.eval().requires_grad_(False)
    return vae


def load_video_to_tensor(
    video_path: str | Path,
    dtype: torch.dtype,
    *,
    frame_start: int = 0,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, VideoMetadata]:
    video_path = Path(video_path)
    if frame_start < 0:
        raise ValueError(f"frame_start must be non-negative, got {frame_start}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    reader = imageio.get_reader(str(video_path))
    frames_cache: list[np.ndarray] | None = None
    try:
        metadata = reader.get_meta_data()
        width, height = metadata["size"]
        fps = float(metadata["fps"])

        try:
            frame_count = _count_frames_from_reader(reader, metadata)
        except RuntimeError:
            frames_cache = [np.asarray(frame, dtype=np.uint8) for frame in reader]
            frame_count = len(frames_cache)

        if frame_count <= 0:
            raise RuntimeError(f"Video {video_path} has no frames.")
        if frame_start >= frame_count:
            raise ValueError(f"frame_start={frame_start} exceeds available frame_count={frame_count} for {video_path}")

        selected_frame_count = frame_count - frame_start
        if max_frames is not None:
            selected_frame_count = min(selected_frame_count, int(max_frames))
        if selected_frame_count <= 0:
            raise RuntimeError(f"No frames remain after slicing {video_path}")

        video = torch.empty((1, 3, selected_frame_count, int(height), int(width)), dtype=dtype, device="cpu")

        frame_iter = enumerate(frames_cache) if frames_cache is not None else enumerate(reader)
        written = 0
        for frame_idx, frame in frame_iter:
            if frame_idx < frame_start:
                continue
            dst_idx = frame_idx - frame_start
            if dst_idx >= selected_frame_count:
                break

            frame = np.asarray(frame, dtype=np.uint8)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                raise ValueError(f"Unexpected frame shape {frame.shape} in {video_path}")
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]

            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous().to(dtype=dtype)
            frame_tensor = frame_tensor.div(127.5).sub(1.0)
            video[0, :, dst_idx].copy_(frame_tensor)
            written += 1

        if written != selected_frame_count:
            video = video[:, :, :written].contiguous()
            selected_frame_count = written
    finally:
        reader.close()

    resolved_metadata = VideoMetadata(
        width=int(width),
        height=int(height),
        fps=fps,
        frame_count=int(selected_frame_count),
    )
    return video, resolved_metadata


def validate_vae_video_size(metadata: VideoMetadata) -> None:
    if metadata.width % 32 != 0 or metadata.height % 32 != 0:
        raise ValueError(
            f"Wan2.2 TI2V VAE expects width/height divisible by 32, got {metadata.width}x{metadata.height}."
        )


def save_latent_payload(
    latents: torch.Tensor,
    save_path: str | Path,
    *,
    video_path: str | Path,
    vae_path: str | Path,
    metadata: VideoMetadata,
    device: str,
    dtype: torch.dtype,
    tiled: bool,
    tile_size: Sequence[int],
    tile_stride: Sequence[int],
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latents": latents.cpu(),
        "video_path": str(Path(video_path).resolve()),
        "vae_path": str(Path(vae_path).resolve()),
        "fps": float(metadata.fps),
        "frame_count": int(metadata.frame_count),
        "video_size": [int(metadata.width), int(metadata.height)],
        "device": device,
        "dtype": str(dtype),
        "tiled": bool(tiled),
        "tile_size": [int(tile_size[0]), int(tile_size[1])],
        "tile_stride": [int(tile_stride[0]), int(tile_stride[1])],
    }
    torch.save(payload, save_path)


def encode_stacked_track_video_to_latents(
    video_path: str | Path,
    vae_path: str | Path,
    latent_output_path: str | Path | None = None,
    *,
    device: str = "auto",
    dtype_name: str = "auto",
    tiled: bool = True,
    tile_size: Sequence[int] = DEFAULT_TILE_SIZE,
    tile_stride: Sequence[int] = DEFAULT_TILE_STRIDE,
    frame_start: int = 0,
    max_frames: int | None = None,
) -> dict[str, Any]:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_torch_dtype(dtype_name, resolved_device)
    video_tensor, metadata = load_video_to_tensor(
        video_path,
        dtype=resolved_dtype,
        frame_start=frame_start,
        max_frames=max_frames,
    )
    validate_vae_video_size(metadata)

    vae = load_wan_ti2v_vae(vae_path, device=resolved_device, dtype=resolved_dtype)
    with torch.inference_mode():
        latents = vae.encode(
            video_tensor,
            device=resolved_device,
            tiled=tiled,
            tile_size=tuple(int(v) for v in tile_size),
            tile_stride=tuple(int(v) for v in tile_stride),
        )
    latents = latents.detach().cpu()

    if latent_output_path is None:
        latent_output_path = build_default_latent_output_path(video_path)
    save_latent_payload(
        latents,
        latent_output_path,
        video_path=video_path,
        vae_path=vae_path,
        metadata=metadata,
        device=resolved_device,
        dtype=resolved_dtype,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )

    if resolved_device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "latent_output_path": str(Path(latent_output_path).resolve()),
        "vae_path": str(Path(vae_path).resolve()),
        "device": resolved_device,
        "dtype": str(resolved_dtype),
        "video_path": str(Path(video_path).resolve()),
        "input_video_size": [metadata.width, metadata.height],
        "input_frame_count": metadata.frame_count,
        "input_fps": metadata.fps,
        "frame_start": int(frame_start),
        "latent_shape": list(latents.shape),
        "latent_dtype": str(latents.dtype),
        "tiled": bool(tiled),
        "tile_size": [int(tile_size[0]), int(tile_size[1])],
        "tile_stride": [int(tile_stride[0]), int(tile_stride[1])],
    }


def main() -> None:
    args = parse_args()

    render_summary: dict[str, Any] | None = None
    video_path = Path(args.input_video) if args.input_video is not None else Path(args.output)

    if args.input_video is None:
        render_summary = render_episode_tracks_to_mp4(
            meta_path=args.meta_path,
            episode_index=args.index,
            output_path=video_path,
            num_points=args.num_points,
            point_radius=args.point_radius,
            seed=args.seed,
            quality=args.quality,
        )

    summaries: dict[str, Any] = {}
    if render_summary is not None:
        summaries["render"] = render_summary

    if not args.skip_latent:
        latent_summary = encode_stacked_track_video_to_latents(
            video_path=video_path,
            vae_path=args.vae_path,
            latent_output_path=args.latent_output,
            device=args.device,
            dtype_name=args.dtype,
            tiled=not args.no_tiled,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            frame_start=args.frame_start,
            max_frames=args.max_frames,
        )
        summaries["latents"] = latent_summary

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
