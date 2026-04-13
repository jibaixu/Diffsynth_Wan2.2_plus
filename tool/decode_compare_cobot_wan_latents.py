#!/usr/bin/env python3
"""Decode WAN VAE latents and compare them against preprocessed source videos."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import imageio.v2 as imageio
import torch
from tqdm import tqdm

from diffsynth.core.data.operators import ToVideoTensor
from examples.wanvideo.model_inference.inference_support import VideoSaver
from tool.extract_cobot_wan_vae_latents import (
    build_video_loader,
    load_jsonl,
    load_wan_vae,
    prepare_video_tensor,
    resolve_compute_dtype,
    resolve_device,
    resolve_under_root,
    validate_device_choice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode WAN VAE latents and save per-view comparison videos."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/linzengrong/Datasets/Cobot_Magic_all/Cobot_Magic_cut_banana"),
        help="Dataset root containing metadata, videos_clipped, and latent files.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("meta/episodes_latents_smoke.jsonl"),
        help="Input latent metadata path, relative to dataset root unless absolute.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("latents_smoke_compare"),
        help="Output root for comparison videos, relative to dataset root unless absolute.",
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=Path("latents_smoke"),
        help="Root prefix used by latent metadata paths.",
    )
    parser.add_argument(
        "--source-video-root",
        type=Path,
        default=Path("videos_clipped"),
        help="Root prefix of source videos relative to dataset root.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/linzengrong/Models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
        help="Local Wan2.2 VAE checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="VAE compute dtype.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=288,
        help="Target fit-box height used during source-video preprocessing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Target fit-box width used during source-video preprocessing.",
    )
    parser.add_argument(
        "--spatial-division-factor",
        type=int,
        default=32,
        help="Height/width divisibility used during source-video preprocessing.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1920 * 1080,
        help="Max pixels for ImageCropAndResize.",
    )
    parser.add_argument(
        "--resize-mode",
        type=str,
        default="fit",
        choices=("fit", "crop"),
        help="Resize mode for ImageCropAndResize.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Override output FPS. Use 0 to inherit FPS from the source mp4.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=5,
        help="Output mp4 quality passed to imageio/ffmpeg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing comparison videos.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N metadata rows; 0 means all.",
    )
    return parser.parse_args()


def validate_loaded_latent(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim != 5:
        raise ValueError(f"Expected latent tensor with shape (1,C,T,H,W), got {tuple(tensor.shape)} at {path}")
    if int(tensor.shape[0]) != 1:
        raise ValueError(f"Expected latent tensor leading dimension 1, got {tuple(tensor.shape)} at {path}")
    return tensor


def decode_latent_video(
    vae,
    latent_tensor: torch.Tensor,
    *,
    device: str,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    latent_tensor = latent_tensor.to(device=device, dtype=compute_dtype)
    decoded = vae.decode(latent_tensor, device=device)
    if not isinstance(decoded, torch.Tensor) or decoded.ndim != 5:
        raise TypeError(f"Expected decoded video tensor with shape (V,C,T,H,W), got {type(decoded).__name__}")
    return decoded.detach().to(dtype=torch.float32).cpu()


def source_fps_from_video(video_path: Path, fps_override: float) -> float:
    if fps_override > 0:
        return float(fps_override)
    reader = imageio.get_reader(str(video_path))
    try:
        metadata = reader.get_meta_data()
    finally:
        reader.close()
    fps = metadata.get("fps")
    if fps is None:
        return 30.0
    return float(fps)


def latent_to_source_video_rel(
    latent_rel: PurePosixPath,
    *,
    latent_root: PurePosixPath,
    source_video_root: PurePosixPath,
) -> PurePosixPath:
    relative_tail = latent_rel.relative_to(latent_root)
    return (source_video_root / relative_tail).with_suffix(".mp4")


def latent_to_output_video_rel(
    latent_rel: PurePosixPath,
    *,
    latent_root: PurePosixPath,
    output_root: Path,
) -> Path:
    relative_tail = latent_rel.relative_to(latent_root)
    return output_root / Path(*relative_tail.parent.parts) / f"{relative_tail.stem}_compare.mp4"


def process_record(
    record: dict,
    *,
    dataset_root: Path,
    latent_root: PurePosixPath,
    output_root: Path,
    source_video_root: PurePosixPath,
    vae,
    loader,
    to_video_tensor: ToVideoTensor,
    device: str,
    compute_dtype: torch.dtype,
    fps_override: float,
    quality: int,
    overwrite: bool,
) -> list[Path]:
    latent_rel_list = record.get("video")
    if not isinstance(latent_rel_list, list) or len(latent_rel_list) == 0:
        raise ValueError("Expected metadata field `video` to be a non-empty list of latent paths.")

    outputs: list[Path] = []
    video_saver: VideoSaver | None = None
    for latent_rel_raw in latent_rel_list:
        latent_rel = PurePosixPath(str(latent_rel_raw))
        if not latent_rel.is_relative_to(latent_root):
            raise ValueError(f"Latent path {latent_rel} is not under latent root {latent_root}.")

        source_video_rel = latent_to_source_video_rel(
            latent_rel,
            latent_root=latent_root,
            source_video_root=source_video_root,
        )
        output_video_rel = latent_to_output_video_rel(
            latent_rel,
            latent_root=latent_root,
            output_root=output_root,
        )

        latent_path = (dataset_root / latent_rel).resolve()
        source_video_path = (dataset_root / source_video_rel).resolve()
        output_video_path = output_video_rel.resolve()
        if output_video_path.exists() and not overwrite:
            print(f"[skip] {latent_path} -> {output_video_path}")
            outputs.append(output_video_path)
            continue
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing latent file: {latent_path}")
        if not source_video_path.is_file():
            raise FileNotFoundError(f"Missing source video: {source_video_path}")

        latent_tensor = validate_loaded_latent(latent_path)
        decoded_video = decode_latent_video(
            vae,
            latent_tensor,
            device=device,
            compute_dtype=compute_dtype,
        )
        original_video = prepare_video_tensor(loader, to_video_tensor, source_video_path, record)
        if not isinstance(original_video, torch.Tensor):
            raise TypeError("Expected source-video preprocessing to return a tensor.")
        original_video = original_video.to(dtype=torch.float32).cpu()

        if decoded_video.ndim != 5 or original_video.ndim != 5:
            raise ValueError(
                f"Expected both videos to be 5D tensors, got decoded={tuple(decoded_video.shape)} "
                f"original={tuple(original_video.shape)}"
            )
        if int(decoded_video.shape[0]) != 1 or int(original_video.shape[0]) != 1:
            raise ValueError(
                f"Expected per-view tensors with V=1, got decoded={tuple(decoded_video.shape)} "
                f"original={tuple(original_video.shape)}"
            )
        if tuple(decoded_video.shape[3:5]) != tuple(original_video.shape[3:5]):
            raise ValueError(
                f"Spatial shape mismatch for {latent_path}: decoded={tuple(decoded_video.shape)} "
                f"original={tuple(original_video.shape)}. These latents were likely extracted with a different "
                f"height/width or spatial_division_factor."
            )

        fps = source_fps_from_video(source_video_path, fps_override)
        if video_saver is None or float(video_saver.fps) != float(fps):
            video_saver = VideoSaver(fps=fps, quality=quality, show_progress=True)

        if int(decoded_video.shape[2]) != int(original_video.shape[2]):
            print(
                f"Frame-count mismatch for {latent_path.name}: "
                f"original={int(original_video.shape[2])}, decoded={int(decoded_video.shape[2])}; using truncate_min."
            )

        print(
            f"[compare] latent={latent_path} "
            f"latent_shape={tuple(int(x) for x in latent_tensor.shape)} "
            f"decoded_vcthw={tuple(int(x) for x in decoded_video.shape)} "
            f"source_vcthw={tuple(int(x) for x in original_video.shape)} "
            f"output={output_video_path}"
        )

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        video_saver.save_comparison(
            original_video,
            decoded_video,
            output_video_path.parent,
            output_video_path.name,
            length_mode="truncate_min",
        )
        outputs.append(output_video_path)
    return outputs


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    metadata_path = resolve_under_root(dataset_root, args.metadata_path).resolve()
    output_root_path = resolve_under_root(dataset_root, args.output_root).resolve()
    model_path = args.model_path.resolve()
    latent_root = PurePosixPath(args.latent_root.as_posix())
    source_video_root = PurePosixPath(args.source_video_root.as_posix())

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = resolve_device(args.device)
    validate_device_choice(args.device, device)
    compute_dtype = resolve_compute_dtype(device, args.compute_dtype)

    print(f"Dataset root: {dataset_root}")
    print(f"Input metadata: {metadata_path}")
    print(f"Output root: {output_root_path}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"Compute dtype: {compute_dtype}")
    print(f"Resize target: {args.height}x{args.width} ({args.resize_mode})")

    rows = load_jsonl(metadata_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No records found in {metadata_path}")

    loader = build_video_loader(args)
    to_video_tensor = ToVideoTensor()
    vae = load_wan_vae(model_path, device=device, compute_dtype=compute_dtype)

    total_outputs = 0
    for record in tqdm(rows, desc="Episodes"):
        outputs = process_record(
            record,
            dataset_root=dataset_root,
            latent_root=latent_root,
            output_root=output_root_path,
            source_video_root=source_video_root,
            vae=vae,
            loader=loader,
            to_video_tensor=to_video_tensor,
            device=device,
            compute_dtype=compute_dtype,
            fps_override=args.fps,
            quality=args.quality,
            overwrite=args.overwrite,
        )
        total_outputs += len(outputs)

    print(f"Wrote comparison videos: {total_outputs}")


if __name__ == "__main__":
    main()
