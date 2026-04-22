#!/usr/bin/env python3
import json
from pathlib import Path

import av
import numpy as np


DATASET_ROOT = Path("/data_jbx/Datasets/RoboTwin2.0_lerobot_v2")
INPUT_JSONL = DATASET_ROOT / "episodes_train_cam_high.filtered.jsonl"
OUTPUT_JSONL = INPUT_JSONL.with_name(f"{INPUT_JSONL.stem}.with_track.jsonl")


def get_single_video_relative_path(record: dict, line_no: int) -> tuple[str, bool]:
    if "video" not in record:
        raise KeyError(f"Missing 'video' at line {line_no}")

    video_field = record["video"]
    if isinstance(video_field, str):
        if not video_field:
            raise ValueError(f"Empty 'video' path at line {line_no}")
        return video_field, False

    if not isinstance(video_field, list):
        raise TypeError(
            f"Unsupported 'video' type at line {line_no}: {type(video_field).__name__}"
        )
    if len(video_field) != 1:
        raise ValueError(
            f"'video' must contain exactly one path at line {line_no}, got {len(video_field)}"
        )

    video_path = video_field[0]
    if not isinstance(video_path, str) or not video_path:
        raise ValueError(f"Invalid 'video' entry at line {line_no}: {video_path!r}")
    return video_path, True


def build_track_relative_path(video_relative_path: str, line_no: int) -> str:
    video_path = Path(video_relative_path)
    parts = video_path.parts
    if len(parts) != 5:
        raise ValueError(
            f"Unexpected video path structure at line {line_no}: {video_relative_path}"
        )

    dataset_name, videos_dir, chunk_name, view_name, filename = parts
    if videos_dir != "videos":
        raise ValueError(
            f"Expected 'videos' directory at line {line_no}: {video_relative_path}"
        )
    if "images" not in view_name:
        raise ValueError(
            f"Expected video view containing 'images' at line {line_no}: {video_relative_path}"
        )
    if Path(filename).suffix.lower() != ".mp4":
        raise ValueError(f"Expected .mp4 video at line {line_no}: {video_relative_path}")

    track_view_name = view_name.replace("images", "tracks")
    return str(
        Path(dataset_name)
        / "tracks"
        / chunk_name
        / track_view_name
        / f"{video_path.stem}.npz"
    )


def resolve_dataset_path(relative_path: str, line_no: int, kind: str) -> Path:
    path = DATASET_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"{kind} file not found at line {line_no}: {path}"
        )
    return path


def load_video_num_frames(video_path: Path) -> int:
    container = av.open(str(video_path))
    try:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {video_path}")
        stream = container.streams.video[0]
        if stream.frames and stream.frames > 0:
            return int(stream.frames)

        frame_count = 0
        for _ in container.decode(video=0):
            frame_count += 1
        if frame_count <= 0:
            raise ValueError(f"Decoded zero frames from {video_path}")
        return frame_count
    finally:
        container.close()


def load_track_num_frames(track_path: Path) -> int:
    with np.load(track_path, allow_pickle=False) as data:
        if "tracks" not in data:
            raise KeyError(f"Missing 'tracks' array in {track_path}")

        tracks = data["tracks"]
        if tracks.ndim < 1:
            raise ValueError(f"'tracks' must have at least 1 dimension in {track_path}")
        track_num_frames = int(tracks.shape[0])
        if track_num_frames <= 0:
            raise ValueError(f"'tracks' has invalid time dimension in {track_path}: {tracks.shape}")

        if "vis" in data:
            vis = data["vis"]
            if vis.ndim < 1:
                raise ValueError(f"'vis' must have at least 1 dimension in {track_path}")
            if int(vis.shape[0]) != track_num_frames:
                raise ValueError(
                    f"'vis' time dimension mismatch in {track_path}: "
                    f"vis.shape[0]={vis.shape[0]}, tracks.shape[0]={track_num_frames}"
                )

        return track_num_frames


def require_optional_int(record: dict, key: str, line_no: int) -> int | None:
    if key not in record:
        return None
    value = record[key]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid '{key}' at line {line_no}: {value!r}") from exc


def process_jsonl(input_path: Path, output_path: Path) -> dict[str, int]:
    total_rows = 0
    validated_rows = 0
    video_length_cache: dict[str, int] = {}
    track_length_cache: dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    try:
        with input_path.open("r", encoding="utf-8") as src, temp_output_path.open(
            "w", encoding="utf-8"
        ) as dst:
            for line_no, raw_line in enumerate(src, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                total_rows += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

                video_relative_path, video_is_list = get_single_video_relative_path(record, line_no)
                track_relative_path = build_track_relative_path(video_relative_path, line_no)

                if video_relative_path not in video_length_cache:
                    video_path = resolve_dataset_path(video_relative_path, line_no, kind="Video")
                    video_length_cache[video_relative_path] = load_video_num_frames(video_path)

                if track_relative_path not in track_length_cache:
                    track_path = resolve_dataset_path(track_relative_path, line_no, kind="Track")
                    track_length_cache[track_relative_path] = load_track_num_frames(track_path)

                video_num_frames = video_length_cache[video_relative_path]
                track_num_frames = track_length_cache[track_relative_path]
                if video_num_frames != track_num_frames:
                    raise ValueError(
                        f"Time dimension mismatch at line {line_no}: "
                        f"video={video_relative_path} ({video_num_frames}), "
                        f"track={track_relative_path} ({track_num_frames})"
                    )

                raw_length = require_optional_int(record, "raw_length", line_no)
                if raw_length is not None and raw_length != video_num_frames:
                    raise ValueError(
                        f"'raw_length' mismatch at line {line_no}: "
                        f"raw_length={raw_length}, video={video_num_frames}, track={track_num_frames}, "
                        f"video_path={video_relative_path}, track_path={track_relative_path}"
                    )

                record["track"] = [track_relative_path] if video_is_list else track_relative_path
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                validated_rows += 1

        temp_output_path.replace(output_path)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise

    return {
        "total_rows": total_rows,
        "validated_rows": validated_rows,
        "unique_videos": len(video_length_cache),
        "unique_tracks": len(track_length_cache),
    }


def main() -> None:
    stats = process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
    print(f"Input JSONL: {INPUT_JSONL}")
    print(f"Output JSONL: {OUTPUT_JSONL}")
    print(f"Total rows: {stats['total_rows']}")
    print(f"Validated rows: {stats['validated_rows']}")
    print(f"Unique videos: {stats['unique_videos']}")
    print(f"Unique tracks: {stats['unique_tracks']}")


if __name__ == "__main__":
    main()
