### filter_across_chunk_clips.py
固定读取 `episodes_train_cam_high.jsonl`，按 global chunk stitching 边界过滤跨边界的 clip，输出过滤后的 jsonl，并打印过滤占比。

### update_track_and_len.py
固定读取 `episodes_train_cam_high.filtered.jsonl`，按 `video` 路径补 `track` 字段，校验 `video`、`track` 与 `raw_length` 的时间维一致，输出新的 jsonl。
 
