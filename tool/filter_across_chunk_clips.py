#!/usr/bin/env python3
import json
from pathlib import Path


INPUT_JSONL = Path("/data_jbx/Datasets/RoboTwin2.0_lerobot_v2/episodes_train_cam_high.jsonl")
OUTPUT_JSONL = INPUT_JSONL.with_name(f"{INPUT_JSONL.stem}.filtered.jsonl")
GLOBAL_CHUNK_SIZE = 200
GLOBAL_CHUNK_OVERLAP = 20


def build_overlapping_chunk_ranges(
    total_frames: int,
    chunk_size: int = GLOBAL_CHUNK_SIZE,
    overlap: int = GLOBAL_CHUNK_OVERLAP,
) -> list[tuple[int, int]]:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, {chunk_size}), got {overlap}")

    ranges: list[tuple[int, int]] = []
    chunk_start = 0
    while chunk_start < total_frames:
        chunk_end = min(chunk_start + chunk_size, total_frames)
        ranges.append((chunk_start, chunk_end))
        if chunk_end == total_frames:
            break
        chunk_start = chunk_end - overlap

    return ranges


def build_stitched_chunk_intervals(
    raw_length: int,
    chunk_size: int = GLOBAL_CHUNK_SIZE,
    overlap: int = GLOBAL_CHUNK_OVERLAP,
) -> list[tuple[int, int]]:
    if raw_length <= 0:
        raise ValueError(f"raw_length must be positive, got {raw_length}")

    chunk_ranges = build_overlapping_chunk_ranges(
        total_frames=raw_length,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if len(chunk_ranges) == 1:
        return [(0, raw_length)]

    intervals: list[tuple[int, int]] = []
    for chunk_index, (chunk_start, _) in enumerate(chunk_ranges):
        if chunk_index + 1 < len(chunk_ranges):
            interval_end = chunk_ranges[chunk_index + 1][0]
        else:
            interval_end = raw_length

        if interval_end > chunk_start:
            intervals.append((chunk_start, interval_end))

    return intervals


def require_int(record: dict, key: str, line_no: int) -> int:
    if key not in record:
        raise KeyError(f"Missing '{key}' at line {line_no}")
    value = record[key]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid '{key}' at line {line_no}: {value!r}") from exc


def validate_clip_record(record: dict, line_no: int) -> tuple[int, int, int]:
    start_frame = require_int(record, "start_frame", line_no)
    end_frame = require_int(record, "end_frame", line_no)
    raw_length = require_int(record, "raw_length", line_no)

    if raw_length <= 0:
        raise ValueError(f"raw_length must be positive at line {line_no}, got {raw_length}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative at line {line_no}, got {start_frame}")
    if end_frame < start_frame:
        raise ValueError(
            f"end_frame must be >= start_frame at line {line_no}, "
            f"got start_frame={start_frame}, end_frame={end_frame}"
        )
    if end_frame >= raw_length:
        raise ValueError(
            f"end_frame must be < raw_length at line {line_no}, "
            f"got end_frame={end_frame}, raw_length={raw_length}"
        )

    if "length" in record:
        length = require_int(record, "length", line_no)
        expected_length = end_frame - start_frame + 1
        if length != expected_length:
            raise ValueError(
                f"length mismatch at line {line_no}: "
                f"length={length}, expected={expected_length}"
            )

    return start_frame, end_frame, raw_length


def clip_is_within_stitched_interval(
    start_frame: int,
    end_frame: int,
    stitched_intervals: list[tuple[int, int]],
) -> bool:
    end_exclusive = end_frame + 1
    # Match the upstream semantics: a clip is safe only if it falls completely
    # inside one stitched interval, not merely outside the explicit overlap span.
    return any(
        interval_start <= start_frame and end_exclusive <= interval_end
        for interval_start, interval_end in stitched_intervals
    )


def filter_jsonl(input_path: Path, output_path: Path) -> dict[str, int]:
    total_rows = 0
    kept_rows = 0
    filtered_rows = 0
    interval_cache: dict[int, list[tuple[int, int]]] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
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

            start_frame, end_frame, raw_length = validate_clip_record(record, line_no)
            if raw_length not in interval_cache:
                interval_cache[raw_length] = build_stitched_chunk_intervals(raw_length)

            stitched_intervals = interval_cache[raw_length]
            if clip_is_within_stitched_interval(start_frame, end_frame, stitched_intervals):
                dst.write(raw_line if raw_line.endswith("\n") else f"{raw_line}\n")
                kept_rows += 1
            else:
                filtered_rows += 1

    return {
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "filtered_rows": filtered_rows,
    }


def main() -> None:
    stats = filter_jsonl(INPUT_JSONL, OUTPUT_JSONL)
    total_rows = stats["total_rows"]
    filtered_rows = stats["filtered_rows"]
    kept_rows = stats["kept_rows"]
    filtered_ratio = 0.0 if total_rows == 0 else filtered_rows / total_rows

    print(f"Input JSONL: {INPUT_JSONL}")
    print(f"Output JSONL: {OUTPUT_JSONL}")
    print(f"GLOBAL_CHUNK_SIZE: {GLOBAL_CHUNK_SIZE}")
    print(f"GLOBAL_CHUNK_OVERLAP: {GLOBAL_CHUNK_OVERLAP}")
    print(f"Total clips: {total_rows}")
    print(f"Kept clips: {kept_rows}")
    print(f"Filtered clips: {filtered_rows}")
    print(f"Filtered ratio: {filtered_ratio:.6f} ({filtered_ratio:.2%})")


if __name__ == "__main__":
    main()
