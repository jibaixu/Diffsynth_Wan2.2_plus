"""
推理引擎：负责所有推理逻辑
包括：检查点管理、数据加载、视频生成
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadTrackMapVideo, OBS_ACTION_NAMES, POSE_NAMES
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.pipelines.wan_video_data import (
    WAN_INFERENCE_DATASET_NUM_FRAMES,
    build_wan_video_dataset,
)

from inference_support import (
    CheckpointPipelineManager,
    DistributedInferenceContext,
    VideoSaver,
    WanInferenceConfig,
    barrier,
    broadcast_object,
    merged_sample_records_path,
    resolve_optional_path,
    suppress_stdout_if,
    write_jsonl_records,
)

_STATE_POSE_INDICES = tuple({name: idx for idx, name in enumerate(OBS_ACTION_NAMES)}[name] for name in POSE_NAMES)


@dataclass(frozen=True)
class InferenceSample:
    sample_index: int
    original_video: torch.Tensor
    action: Optional[object]
    track_video: Optional[torch.Tensor]
    metadata_entry: Dict
    prompt: Optional[str]
    prompt_emb: Optional[object]
    atm_task_emb: Optional[torch.Tensor]
    negative_prompt_emb: Optional[object]
    episode_index: int
    num_views: int
    total_frames: int
    input_height: int
    input_width: int


class InferenceEngine:
    """
    推理引擎：负责整个推理流程

    核心流程：
    1. 加载数据集并选择样本
    2. 发现所有 checkpoint
    3. 初始化 pipeline（只加载一次 VAE/CLIP/T5）
    4. 遍历每个 checkpoint：
       - 更新 DiT/Action Encoder 权重
       - 生成所有样本的视频
    5. 输出总结报告
    """

    def __init__(
        self,
        config: WanInferenceConfig,
        logger,
        dist_context: Optional[DistributedInferenceContext] = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.dist_context = dist_context or DistributedInferenceContext()
        self.video_saver = VideoSaver(
            fps=config.fps,
            quality=config.quality,
            show_progress=self._is_main_process(),
        )
        self.pipeline_manager = CheckpointPipelineManager(
            config,
            logger,
            device=self.dist_context.device,
            verbose=self._is_main_process(),
        )
        self.dataset: Optional[UnifiedDataset] = None
        self.pipeline: Optional[WanVideoPipeline] = None
        self.sample_indices: List[int] = []
        self.checkpoints: List[Path] = []
        self.total_checkpoint_runs: int = 0
        self.num_views: Optional[int] = None
        self.generated_sample_records: List[Dict] = []
        self.atm_engine = None
        self._state_pose_stats: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._state_pose_cache: Dict[str, np.ndarray] = {}
        self._active_output_dirs: Optional[Dict[str, Path]] = None
        self._diagnostic_track_exports: set[tuple[int, int]] = set()

    def run(self) -> None:
        try:
            self._prepare()
            with suppress_stdout_if(not self._is_main_process()):
                self.pipeline = self.pipeline_manager.initialize_pipeline(self.checkpoints)

            checkpoints_to_run: List[Optional[Path]] = self.checkpoints if self.checkpoints else [None]
            self.total_checkpoint_runs = len(checkpoints_to_run)
            summaries: List[Dict] = []
            for ckpt_idx, checkpoint in enumerate(checkpoints_to_run, 1):
                summaries.append(
                    self._process_checkpoint(
                        checkpoint,
                        ckpt_idx=ckpt_idx,
                        total_checkpoints=self.total_checkpoint_runs,
                    )
                )
        except Exception as exc:  # pragma: no cover - runtime protection
            self.logger.error(f"Inference failed: {exc}", exc_info=True)
            raise

    def _prepare(self) -> None:
        self._load_dataset()
        all_sample_indices = list(range(len(self.dataset)))
        self.sample_indices = all_sample_indices[self.dist_context.rank :: self.dist_context.world_size]
        self.checkpoints = self.pipeline_manager.discover_checkpoints()

        if self._is_main_process():
            self.logger.info("Total samples: %s", len(self.dataset))
            if self.checkpoints:
                self.logger.info("Checkpoints to process: %s", len(self.checkpoints))
            else:
                self.logger.info("Checkpoints to process: pretrained-only (1 run)")
            self.logger.info("Resume generation: %s", "enabled" if self._resume_enabled() else "disabled")
            self.logger.info("")

    def _use_autoregressive_history_template_mode(self) -> bool:
        return (
            int(getattr(self.config, "history_template_sampling", 0)) == 1
            and int(getattr(self.config, "num_history_frames", 1)) > 1
        )

    def _load_dataset(self) -> None:
        if self._is_main_process():
            self.logger.info("Loading dataset...")
            if "action" in self.config.data_file_keys:
                self.logger.info("Action stat path: %s", self.config.action_stat_path)
            if self._should_use_atm_track():
                self.logger.info("ATM track prediction enabled; dataset track files will be skipped.")
            if self._use_autoregressive_history_template_mode():
                self.logger.info(
                    "Autoregressive history-template inference enabled; overriding dataset history_template_sampling to 0 to load full episodes."
                )

        with suppress_stdout_if(not self._is_main_process()):
            spatial_division_factor = int(getattr(self.config, "spatial_division_factor", 16))
            dataset_history_template_sampling = (
                0
                if self._use_autoregressive_history_template_mode()
                else int(getattr(self.config, "history_template_sampling", 0))
            )
            dataset_data_file_keys = self._dataset_data_file_keys()
            self.dataset = build_wan_video_dataset(
                self.config.runtime,
                base_path=self.config.dataset_base_path,
                metadata_path=self.config.dataset_metadata_path,
                height=self.config.height,
                width=self.config.width,
                num_frames=int(getattr(self.config, "num_frames", WAN_INFERENCE_DATASET_NUM_FRAMES)),
                num_history_frames=int(getattr(self.config, "num_history_frames", 1)),
                repeat=1,
                resize_mode=self.config.resize_mode,
                max_pixels=self.config.max_pixels,
                data_file_keys=dataset_data_file_keys,
                dataset_num_frames=WAN_INFERENCE_DATASET_NUM_FRAMES,
                action_stat_path=self.config.action_stat_path,
                action_type=self.config.action_type,
                history_template_sampling=dataset_history_template_sampling,
                height_division_factor=spatial_division_factor,
                width_division_factor=spatial_division_factor,
            )

    def _dataset_data_file_keys(self) -> List[str]:
        keys = [str(key).strip() for key in self.config.data_file_keys if str(key).strip()]
        if self._should_use_atm_track():
            keys = [key for key in keys if key != "track"]
        return keys

    def _diagnose_inference(self) -> bool:
        return bool(int(getattr(self.config, "diagnose_inference", 0)))

    def _diagnostic_max_total_frames(self) -> int:
        return max(0, int(getattr(self.config, "diagnostic_max_total_frames", 0)))

    def _diagnostic_export_track_videos(self) -> bool:
        return self._diagnose_inference() and bool(int(getattr(self.config, "diagnostic_export_track_videos", 1)))

    def _process_checkpoint(
        self,
        checkpoint: Optional[Path],
        ckpt_idx: int,
        total_checkpoints: int,
    ) -> Dict:
        checkpoint_name = checkpoint.name if checkpoint is not None else "pretrained"
        if self._is_main_process():
            self.logger.info("\n%s", "#" * 80)
            self.logger.info(
                "Processing checkpoint %s/%s: %s",
                ckpt_idx,
                total_checkpoints,
                checkpoint_name,
            )
            self.logger.info("%s\n", "#" * 80)

        with suppress_stdout_if(not self._is_main_process()):
            self.pipeline_manager.update_checkpoint(checkpoint)
            self.pipeline_manager.prepare_generation_models()
        output_dirs = self._create_output_dirs(checkpoint)
        self.generated_sample_records = []
        self._generate_all_videos(output_dirs, ckpt_idx)
        merged_sample_records = self._finalize_sample_records(output_dirs["root"])
        barrier(self.dist_context)
        with suppress_stdout_if(not self._is_main_process()):
            self.pipeline_manager.release_generation_models()
        barrier(self.dist_context)

        metrics_report = None
        metrics_output_path = output_dirs["root"] / "metrics.json"
        if self.dist_context.is_main_process:
            metrics_report = self._evaluate_generated_outputs(
                output_root=output_dirs["root"],
                checkpoint_name=checkpoint_name,
                sample_records=merged_sample_records,
                metrics_output_path=metrics_output_path,
            )
        barrier(self.dist_context)

        if self.dist_context.is_main_process:
            self.logger.info("\n%s", "=" * 60)
            self.logger.info("VIDEO GENERATION COMPLETED")
            self.logger.info("%s", "=" * 60)
            self.logger.info("Total videos: %s", len(merged_sample_records))
            self.logger.info("Metrics report: %s", metrics_output_path.resolve())
            self.logger.info("%s\n", "=" * 60)
        summary = {
            "checkpoint": checkpoint_name,
            "output_dir": output_dirs["root"].resolve(),
            "num_videos": len(merged_sample_records) if self.dist_context.is_main_process else len(self.sample_indices),
        }
        if metrics_report is not None:
            summary["metrics_path"] = str(metrics_output_path.resolve())
            summary["overall_metrics"] = metrics_report.get("overall", {})
        return summary

    def _evaluate_generated_outputs(
        self,
        *,
        output_root: Path,
        checkpoint_name: str,
        sample_records: List[Dict],
        metrics_output_path: Path,
    ) -> Dict:
        from diffsynth.core.metric.metric import evaluate_and_write_report

        metrics_mode = str(getattr(self.config, "metrics", "core"))
        num_views = int(self.num_views) if self.num_views is not None else 3
        frame_chunk_size = int(getattr(self.config, "num_frames", WAN_INFERENCE_DATASET_NUM_FRAMES))
        batch_videos = int(getattr(self.config, "batch_videos", 1))

        self.logger.info("\n%s", "-" * 60)
        self.logger.info("Starting evaluation: metrics=%s, batch_videos=%s", metrics_mode, batch_videos)
        self.logger.info("%s", "-" * 60)
        report = evaluate_and_write_report(
            output_root=str(output_root),
            checkpoint_name=checkpoint_name,
            metrics_output_path=metrics_output_path,
            sample_records=sample_records,
            num_workers=int(getattr(self.config, "dataset_num_workers", 64)),
            num_views=num_views,
            frame_chunk_size=frame_chunk_size,
            batch_videos=batch_videos,
            metrics=metrics_mode,
        )
        overall_metrics = report.get("overall", {})
        if overall_metrics:
            summary = ", ".join(
                f"{metric_name}={float(metric_value):.4f}"
                for metric_name, metric_value in overall_metrics.items()
            )
            self.logger.info("Overall metrics: %s", summary)
        self.logger.info("Saved metrics report: %s", metrics_output_path.resolve())
        return report

    def _create_output_dirs(self, checkpoint: Optional[Path]) -> Dict[str, Path]:
        output_dir_value = None
        if self.dist_context.is_main_process:
            if checkpoint is None:
                output_dir_value = str(Path("Ckpt") / "pretrained")
            else:
                output_dir_value = str(checkpoint.parent)
        output_dir = Path(broadcast_object(self.dist_context, output_dir_value))
        output_dir.mkdir(parents=True, exist_ok=True)
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir = output_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        if self.dist_context.is_main_process:
            self._save_config(output_dir)
        barrier(self.dist_context)
        if self._is_main_process():
            self.logger.info("Output directory: %s\n", output_dir.resolve())
        return {"root": output_dir, "videos": videos_dir, "diagnostics": diagnostics_dir}

    def _save_config(self, output_dir: Path) -> None:
        config_path = output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            payload = self.config.grouped_config or self.config.values
            json.dump(payload, f, indent=2, ensure_ascii=True)

    def _generate_all_videos(self, output_dirs: Dict[str, Path], ckpt_idx: int) -> None:
        for index, sample_idx in enumerate(self.sample_indices, 1):
            if self._is_main_process():
                self.logger.info("\n%s", "#" * 60)
                if self.total_checkpoint_runs > 1:
                    self.logger.info(
                        "[Checkpoint %s/%s] Video %s/%s (Sample %s)",
                        ckpt_idx,
                        self.total_checkpoint_runs,
                        index,
                        len(self.sample_indices),
                        sample_idx,
                    )
                else:
                    self.logger.info("Video %s/%s (Sample %s)", index, len(self.sample_indices), sample_idx)
                self.logger.info("%s", "#" * 60)

            raw_sample = self.dataset[sample_idx]
            sample = self._prepare_sample(sample_idx, raw_sample)
            self._generate_single_video(sample, output_dirs)

    def _prepare_sample(self, sample_idx: int, sample: Dict) -> InferenceSample:
        original_video = sample["video"]
        if not isinstance(original_video, torch.Tensor) or original_video.ndim != 5:
            raise TypeError(
                f"`sample['video']` must be torch.Tensor with shape (V,C,T,H,W), got {type(original_video)}"
            )
        original_video = original_video.detach().cpu()
        action = sample.get("action")
        if getattr(self.pipeline, "action_injection_mode", "off") == "off":
            action = None
        dataset_track_video = sample.get("track")
        if dataset_track_video is not None:
            if not isinstance(dataset_track_video, torch.Tensor) or dataset_track_video.ndim != 5:
                raise TypeError(
                    f"`sample['track']` must be torch.Tensor with shape (V,C,T,H,W), got {type(dataset_track_video)}"
                )
            dataset_track_video = dataset_track_video.detach().cpu()
        metadata_entry = self._get_metadata_entry(sample_idx)
        episode_index = int(sample["episode_index"])
        prompt = sample.get("prompt")
        prompt_emb = resolve_optional_path(sample.get("prompt_emb"), self.config.dataset_base_path)
        prompt_emb_bert = self._resolve_prompt_emb_bert_path(metadata_entry, prompt_emb)
        negative_prompt_emb = sample.get("negative_prompt_emb", self.config.negative_prompt_emb)
        negative_prompt_emb = resolve_optional_path(negative_prompt_emb, self.config.dataset_base_path)

        num_views = int(original_video.shape[0])
        total_frames = int(original_video.shape[2])
        diagnostic_max_total_frames = self._diagnostic_max_total_frames()
        if diagnostic_max_total_frames > 0:
            capped_total_frames = min(total_frames, diagnostic_max_total_frames)
            if capped_total_frames < total_frames and self._is_main_process():
                self.logger.info(
                    "Diagnostic frame cap enabled for sample %s: total_frames %s -> %s",
                    sample_idx,
                    total_frames,
                    capped_total_frames,
                )
            total_frames = capped_total_frames
        input_height = int(original_video.shape[-2])
        input_width = int(original_video.shape[-1])
        atm_task_emb = None
        track_video = dataset_track_video
        if self._should_use_atm_track():
            if prompt_emb_bert in (None, ""):
                raise FileNotFoundError(
                    "ATM track prediction requires a BERT prompt embedding path; failed to resolve it from `prompt_emb`."
                )
            atm_task_emb = self._load_prompt_emb_bert_tensor(prompt_emb_bert)
            # History-template inference still expects a prebuilt track video sequence.
            if self._use_autoregressive_history_template_mode():
                track_video = self._predict_track_video_for_window(
                    sample=sample_idx,
                    metadata_entry=metadata_entry,
                    input_video=original_video[:, :, :1],
                    task_emb=atm_task_emb,
                    chunk_start=0,
                    target_frames=min(total_frames, int(self._get_atm_engine().num_track_ts)),
                    infer_frames=min(total_frames, int(self._get_atm_engine().num_track_ts)),
                    num_views=num_views,
                    height=input_height,
                    width=input_width,
                )
            else:
                track_video = None
        if self.num_views is None:
            self.num_views = num_views
        elif self.num_views != num_views:
            self.logger.warning("Mixed num_views detected: %s -> %s", self.num_views, num_views)

        if self._diagnose_inference():
            prefix = f"[Diag sample {sample_idx}]"
            self._log_diagnostic_tensor("sample_action", action, prefix=prefix)
            self._log_diagnostic_tensor("sample_dataset_track", dataset_track_video, prefix=prefix)
            self._log_diagnostic_tensor("sample_atm_task_emb", atm_task_emb, prefix=prefix)

        return InferenceSample(
            sample_index=int(sample_idx),
            original_video=original_video,
            action=action,
            track_video=track_video,
            metadata_entry=metadata_entry,
            prompt=prompt,
            prompt_emb=prompt_emb,
            atm_task_emb=atm_task_emb,
            negative_prompt_emb=negative_prompt_emb,
            episode_index=episode_index,
            num_views=num_views,
            total_frames=total_frames,
            input_height=input_height,
            input_width=input_width,
        )

    def _get_metadata_entry(self, sample_idx: int) -> Dict:
        if self.dataset is None or not hasattr(self.dataset, "data"):
            raise RuntimeError("Dataset metadata is unavailable for ATM conditioning.")
        entry = self.dataset.data[sample_idx]
        if not isinstance(entry, dict):
            raise TypeError(f"Expected dataset metadata entry to be dict, got {type(entry)}")
        return entry.copy()

    def _should_use_atm_track(self) -> bool:
        return bool(int(getattr(self.config, "use_atm_track", 1))) and bool(
            getattr(self.config.runtime, "track_context_enabled", False)
        )

    def _resolve_prompt_emb_bert_path(
        self,
        metadata_entry: Dict,
        prompt_emb,
    ) -> Optional[str]:
        for key in ("prompt_emb_bert", "prompt_embed_bert"):
            resolved = resolve_optional_path(metadata_entry.get(key), self.config.dataset_base_path)
            if resolved:
                return resolved
        if prompt_emb in (None, ""):
            return None
        prompt_emb_path = Path(str(prompt_emb))
        if prompt_emb_path.parent.name == "bert":
            return str(prompt_emb_path)
        return str(prompt_emb_path.parent / "bert" / prompt_emb_path.name)

    def _get_atm_engine(self):
        if not self._should_use_atm_track():
            return None
        if self.atm_engine is None:
            from atm.atm_inference import ATMInference

            checkpoint_path = getattr(self.config, "atm_ckpt_path", None)
            if not checkpoint_path:
                raise ValueError("`--atm_ckpt_path` is required when ATM track prediction is enabled.")
            checkpoint_path = str(Path(str(checkpoint_path)).expanduser())
            self.atm_engine = ATMInference(checkpoint_path, device=self.dist_context.device)
            if self._is_main_process():
                self.logger.info(
                    "Loaded ATM checkpoint: %s (num_track_ts=%s, num_track_ids=%s, img_size=%s)",
                    checkpoint_path,
                    self.atm_engine.num_track_ts,
                    self.atm_engine.num_track_ids,
                    self.atm_engine.img_size,
                )
        return self.atm_engine

    def _predict_track_video_for_window(
        self,
        *,
        sample,
        metadata_entry: Dict,
        input_video: torch.Tensor,
        task_emb: torch.Tensor,
        chunk_start: int,
        target_frames: int,
        infer_frames: int,
        num_views: int,
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        atm_engine = self._get_atm_engine()
        if atm_engine is None:
            return None
        if task_emb is None:
            raise ValueError("ATM track prediction requires a loaded task embedding tensor.")
        state_pose = self._load_state_pose_condition(
            metadata_entry,
            atm_engine.num_track_ts,
            start_offset=chunk_start,
        )
        prefix = f"[Diag sample {sample} chunk {chunk_start}]"
        self._log_diagnostic_tensor("atm_state_pose", state_pose, prefix=prefix)
        self._log_diagnostic_tensor("atm_task_emb", task_emb, prefix=prefix)
        predicted_tracks = []
        for view_idx in range(int(num_views)):
            atm_video = self._build_atm_input_video(input_video, atm_engine.img_size, view_idx=view_idx)
            if self._is_main_process():
                self.logger.info(
                    "ATM conditioning sample %s chunk_start %s view %s: video=%s, state_pose=%s, task_emb=%s",
                    sample,
                    chunk_start,
                    view_idx,
                    tuple(atm_video.shape),
                    tuple(state_pose.shape),
                    tuple(task_emb.shape),
                )

            predicted_track = atm_engine.infer(
                atm_video,
                task_emb,
                state_pose,
                track=None,
            )
            predicted_track = predicted_track.detach().to(dtype=torch.float32).cpu()
            if predicted_track.ndim != 4 or predicted_track.shape[0] != 1:
                raise ValueError(
                    f"ATM output must have shape (1,T,N,2), got {tuple(predicted_track.shape)}."
                )
            if self._is_main_process():
                self.logger.info(
                    "ATM predicted track sample %s chunk_start %s view %s: shape=%s, dtype=%s",
                    sample,
                    chunk_start,
                    view_idx,
                    tuple(predicted_track.shape),
                    str(predicted_track.dtype),
                )
            predicted_tracks.append(predicted_track[0])

        if len(predicted_tracks) != int(num_views):
            raise RuntimeError(
                f"Expected {num_views} ATM predictions for sample {sample}, got {len(predicted_tracks)}."
            )
        predicted_tracks = torch.stack(predicted_tracks, dim=0)
        rendered_track = self._render_track_video_from_prediction(
            predicted_tracks,
            height=height,
            width=width,
        )
        self._log_diagnostic_tensor("atm_rendered_track", rendered_track, prefix=prefix)
        return self._slice_and_pad_track_video(
            track_video=rendered_track,
            start=0,
            target_frames=target_frames,
            infer_frames=infer_frames,
        )

    def _load_prompt_emb_bert_tensor(self, path: str) -> torch.Tensor:
        resolved_path = Path(path)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"ATM prompt_emb_bert file not found: {resolved_path}")
        task_emb = torch.load(resolved_path, map_location="cpu")
        if not isinstance(task_emb, torch.Tensor):
            task_emb = torch.as_tensor(task_emb)
        task_emb = task_emb.detach().cpu().float()
        if task_emb.ndim == 1:
            task_emb = task_emb.unsqueeze(0)
        elif task_emb.ndim == 2 and task_emb.shape[0] != 1:
            raise ValueError(f"ATM task embedding must have batch size 1, got shape {tuple(task_emb.shape)}.")
        elif task_emb.ndim != 2:
            raise ValueError(f"ATM task embedding must have shape (1,E) or (E,), got {tuple(task_emb.shape)}.")
        return task_emb

    def _load_state_pose_condition(
        self,
        metadata_entry: Dict,
        num_track_ts: int,
        start_offset: int = 0,
    ) -> torch.Tensor:
        parquet_rel = metadata_entry.get("action")
        if not parquet_rel:
            raise KeyError("Missing `action` parquet path in metadata entry for ATM state_pose loading.")
        parquet_path = resolve_optional_path(parquet_rel, self.config.dataset_base_path)
        start_frame = int(metadata_entry.get("start_frame", 0)) + int(start_offset)
        if parquet_path is None:
            raise FileNotFoundError("Failed to resolve metadata action parquet path for ATM state_pose loading.")

        actions_all = self._load_state_pose_array(parquet_path)
        end_frame = min(start_frame + int(num_track_ts), int(actions_all.shape[0]))
        if end_frame <= start_frame:
            actions = torch.zeros((0, len(_STATE_POSE_INDICES)), dtype=torch.float32)
        else:
            actions = torch.from_numpy(actions_all[start_frame:end_frame][:, _STATE_POSE_INDICES]).float()

        state_min, state_max = self._get_state_pose_stats()
        if actions.numel() > 0:
            actions = self._normalize_bound(actions, state_min, state_max)

        pad_len = int(num_track_ts) - int(actions.shape[0])
        if pad_len > 0:
            pad = torch.zeros((pad_len, state_min.numel()), dtype=actions.dtype if actions.numel() > 0 else torch.float32)
            actions = torch.cat([actions, pad], dim=0)
        return actions.unsqueeze(0)

    def _load_state_pose_array(self, parquet_path: str) -> np.ndarray:
        cached = self._state_pose_cache.get(parquet_path)
        if cached is not None:
            return cached
        state_series = pd.read_parquet(parquet_path, columns=["observation.state"])["observation.state"]
        actions_all = np.stack(state_series.values).astype(np.float32)
        self._state_pose_cache[parquet_path] = actions_all
        return actions_all

    @staticmethod
    def _tensor_stats_message(name: str, value) -> str:
        if value is None:
            return f"{name}=None"
        if isinstance(value, torch.Tensor):
            array = value.detach().to(dtype=torch.float32).cpu().numpy()
        else:
            array = np.asarray(value)
        shape = tuple(array.shape)
        dtype = str(array.dtype)
        if array.size == 0:
            return f"{name}: shape={shape}, dtype={dtype}, size=0"
        finite_mask = np.isfinite(array)
        finite_ratio = float(finite_mask.mean())
        if not finite_mask.any():
            return f"{name}: shape={shape}, dtype={dtype}, finite_ratio={finite_ratio:.6f}, no_finite_values"
        finite_values = array[finite_mask].astype(np.float64, copy=False)
        return (
            f"{name}: shape={shape}, dtype={dtype}, finite_ratio={finite_ratio:.6f}, "
            f"min={finite_values.min():.6f}, max={finite_values.max():.6f}, "
            f"mean={finite_values.mean():.6f}, std={finite_values.std():.6f}"
        )

    def _log_diagnostic_tensor(self, name: str, value, *, prefix: str = "") -> None:
        if not self._diagnose_inference() or not self._is_main_process():
            return
        message = self._tensor_stats_message(name, value)
        if prefix:
            self.logger.info("%s %s", prefix, message)
        else:
            self.logger.info("%s", message)

    def _load_dataset_track_video_for_window(
        self,
        *,
        metadata_entry: Dict,
        chunk_start: int,
        target_frames: int,
        infer_frames: int,
        height: int,
        width: int,
    ) -> Optional[torch.Tensor]:
        track_data = metadata_entry.get("track")
        if track_data in (None, ""):
            return None
        absolute_start = int(metadata_entry.get("start_frame", 0)) + int(chunk_start)
        frame_indices = [absolute_start + frame_id for frame_id in range(int(target_frames))]
        renderer = LoadTrackMapVideo(
            base_path=self.config.dataset_base_path,
            height=height,
            width=width,
            num_frames=max(1, int(target_frames)),
            time_division_factor=4,
            time_division_remainder=1,
            num_points=int(getattr(self.config, "track_num_points", 256)),
            point_radius=int(getattr(self.config, "track_point_radius", 6)),
            seed=int(getattr(self.config, "track_seed", 42)),
            apply_noise=False,
            noise_std=0.0,
        )
        track_video = renderer({"data": track_data, "frame_indices": frame_indices})
        return self._slice_and_pad_track_video(
            track_video=track_video,
            start=0,
            target_frames=target_frames,
            infer_frames=infer_frames,
        )

    def _maybe_export_diagnostic_track_comparison(
        self,
        *,
        sample: InferenceSample,
        chunk_start: int,
        chunk_track: Optional[torch.Tensor],
        target_frames: int,
        infer_frames: int,
    ) -> None:
        if (
            not self._diagnostic_export_track_videos()
            or not self._is_main_process()
            or self._active_output_dirs is None
            or chunk_track is None
            or chunk_start != 0
        ):
            return
        export_key = (int(sample.sample_index), int(chunk_start))
        if export_key in self._diagnostic_track_exports:
            return
        self._diagnostic_track_exports.add(export_key)

        gt_track = self._load_dataset_track_video_for_window(
            metadata_entry=sample.metadata_entry,
            chunk_start=chunk_start,
            target_frames=target_frames,
            infer_frames=infer_frames,
            height=sample.input_height,
            width=sample.input_width,
        )
        prefix = f"[Diag sample {sample.sample_index} chunk {chunk_start}]"
        self._log_diagnostic_tensor("atm_chunk_track", chunk_track, prefix=prefix)
        self._log_diagnostic_tensor("dataset_chunk_track", gt_track, prefix=prefix)
        if gt_track is None:
            self.logger.info("%s dataset track is unavailable; skipping diagnostic track comparison export.", prefix)
            return
        diagnostics_dir = self._active_output_dirs.get("diagnostics")
        if diagnostics_dir is None:
            return
        compare_name = f"track_compare_s{sample.sample_index:06d}_ep{sample.episode_index}_chunk{chunk_start:04d}.mp4"
        saved_path = self.video_saver.save_comparison(
            gt_track,
            chunk_track,
            diagnostics_dir,
            compare_name,
        )
        self.logger.info("%s saved track comparison: %s", prefix, saved_path.resolve())

    def _log_diagnostic_chunk_state(
        self,
        *,
        sample: InferenceSample,
        chunk_start: int,
        input_video,
        action,
        track,
        predicted_video=None,
    ) -> None:
        if not self._diagnose_inference() or not self._is_main_process():
            return
        prefix = f"[Diag sample {sample.sample_index} chunk {chunk_start}]"
        self._log_diagnostic_tensor("chunk_input_video", input_video, prefix=prefix)
        self._log_diagnostic_tensor("chunk_action", action, prefix=prefix)
        self._log_diagnostic_tensor("chunk_track", track, prefix=prefix)
        if predicted_video is not None:
            self._log_diagnostic_tensor("chunk_video", predicted_video, prefix=prefix)

    def _get_state_pose_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._state_pose_stats is None:
            stat_path = getattr(self.config, "action_stat_path", None)
            if not stat_path:
                raise ValueError("`--action_stat_path` is required for ATM state_pose normalization.")
            with open(stat_path, "r", encoding="utf-8") as file_handle:
                stats = json.load(file_handle)
            state_pose_stats = stats.get("state_pose")
            if not isinstance(state_pose_stats, dict):
                raise KeyError(f"Missing `state_pose` statistics in {stat_path}.")
            data_min = torch.as_tensor(state_pose_stats.get("p01"), dtype=torch.float32)
            data_max = torch.as_tensor(state_pose_stats.get("p99"), dtype=torch.float32)
            if data_min.numel() != len(_STATE_POSE_INDICES) or data_max.numel() != len(_STATE_POSE_INDICES):
                raise ValueError(
                    f"Expected 14-dim state_pose stats, got {data_min.numel()} and {data_max.numel()} from {stat_path}."
                )
            self._state_pose_stats = (data_min, data_max)
        return self._state_pose_stats

    @staticmethod
    def _normalize_bound(
        data: torch.Tensor,
        data_min: torch.Tensor,
        data_max: torch.Tensor,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        normalized = 2.0 * (data - data_min) / (data_max - data_min + eps) - 1.0
        return torch.clamp(normalized, min=clip_min, max=clip_max)

    def _build_atm_input_video(
        self,
        original_video: torch.Tensor,
        img_size: tuple[int, int],
        *,
        view_idx: int,
    ) -> torch.Tensor:
        if not isinstance(original_video, torch.Tensor) or original_video.ndim != 5:
            raise TypeError("ATM input video expects `original_video` with shape (V,C,T,H,W).")
        if view_idx < 0 or view_idx >= int(original_video.shape[0]):
            raise IndexError(f"`view_idx`={view_idx} is out of range for {int(original_video.shape[0])} views.")

        frame = original_video[view_idx : view_idx + 1, :, :1].detach().to(dtype=torch.float32)
        frame = frame.permute(0, 2, 1, 3, 4).contiguous()
        frame = torch.clamp((frame + 1.0) * 127.5, 0.0, 255.0)
        target_height, target_width = int(img_size[0]), int(img_size[1])
        if tuple(frame.shape[-2:]) != (target_height, target_width):
            flat = frame.reshape(-1, int(frame.shape[2]), int(frame.shape[3]), int(frame.shape[4]))
            flat = F.interpolate(flat, size=(target_height, target_width), mode="bilinear", align_corners=False)
            frame = flat.reshape(1, 1, int(flat.shape[1]), target_height, target_width)
        return frame.contiguous()

    def _render_track_video_from_prediction(
        self,
        predicted_track: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if predicted_track.ndim != 4 or predicted_track.shape[-1] != 2:
            raise ValueError(f"Expected predicted_track with shape (V,T,N,2), got {tuple(predicted_track.shape)}.")

        track_np = predicted_track.detach().to(dtype=torch.float32).cpu().numpy()
        view_videos = []
        for view_idx in range(int(track_np.shape[0])):
            view_track = track_np[view_idx]
            vis = np.isfinite(view_track).all(axis=-1)
            renderer = LoadTrackMapVideo(
                base_path=self.config.dataset_base_path,
                height=height,
                width=width,
                num_frames=int(view_track.shape[0]),
                time_division_factor=4,
                time_division_remainder=1,
                num_points=min(
                    int(getattr(self.config, "track_num_points", view_track.shape[1])),
                    int(view_track.shape[1]),
                ),
                point_radius=int(getattr(self.config, "track_point_radius", 6)),
                seed=int(getattr(self.config, "track_seed", 42)) + view_idx,
                # Training injects track noise to simulate ATM errors.
                # Inference should render the predicted tracks directly.
                apply_noise=False,
                noise_std=0.0,
            )
            visible_point_indices = np.flatnonzero(np.any(vis, axis=0))
            if visible_point_indices.size == 0:
                visible_point_indices = np.arange(view_track.shape[1], dtype=np.int64)
            sample_size = min(renderer.num_points, visible_point_indices.size)
            rng = np.random.default_rng(renderer.seed)
            point_indices = np.asarray(
                rng.choice(visible_point_indices, size=sample_size, replace=False),
                dtype=np.int64,
            )
            colors = renderer._generate_distinct_colors(len(point_indices))
            noise_seed = int(renderer.seed) + 1000003

            rendered_frames = []
            for frame_id in range(view_track.shape[0]):
                frame = renderer._render_frame(
                    view_track,
                    vis,
                    point_indices,
                    colors,
                    frame_id,
                    noise_seed=noise_seed,
                )
                rendered_frames.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous())
            view_video = torch.stack(rendered_frames, dim=1).float()
            view_videos.append(view_video)

        video = torch.stack(view_videos, dim=0)
        return video * (2.0 / 255.0) - 1.0

    @staticmethod
    def _build_video_name(sample: InferenceSample) -> str:
        return f"video_s{sample.sample_index:06d}_ep{sample.episode_index}.mp4"

    def _resume_enabled(self) -> bool:
        return bool(int(getattr(self.config, "resume", 1)))

    def _build_video_output_path(
        self,
        sample: InferenceSample,
        output_dirs: Dict[str, Path],
    ) -> Path:
        return output_dirs["videos"] / self._build_video_name(sample)

    def _generate_single_video(
        self,
        sample: InferenceSample,
        output_dirs: Dict[str, Path],
    ) -> None:
        saved_path = self._build_video_output_path(sample, output_dirs)
        if self._resume_enabled() and saved_path.is_file():
            self._record_generated_sample(
                video_path=saved_path,
                prompt=sample.prompt,
                episode_index=sample.episode_index,
                sample_index=sample.sample_index,
            )
            if self._is_main_process():
                self.logger.info("Skipped existing: %s\n", saved_path.resolve())
            return
        if saved_path.is_file() and self._is_main_process():
            self.logger.info("Overwriting existing video because resume=0: %s", saved_path.resolve())

        self._active_output_dirs = output_dirs
        self._log_sample(sample)
        predicted_video = self._predict_video(sample)
        self._active_output_dirs = None
        predicted_video = self._finalize_predicted_video(predicted_video, sample.original_video)

        saved_path = self.video_saver.save_comparison(
            sample.original_video[:, :, : int(predicted_video.shape[2])],
            predicted_video,
            output_dirs["videos"],
            saved_path.name,
        )
        self._record_generated_sample(
            video_path=saved_path,
            prompt=sample.prompt,
            episode_index=sample.episode_index,
            sample_index=sample.sample_index,
        )
        if self._is_main_process():
            self.logger.info("Saved: %s\n", saved_path.resolve())

    def _log_sample(self, sample: InferenceSample) -> None:
        if not self._is_main_process():
            return
        self.logger.info("Episode: %s", sample.episode_index)
        self.logger.info("Loaded shape: %s", tuple(sample.original_video.shape))
        self.logger.info("Generating with diffusion (frames: %s)", sample.total_frames)
        self.logger.info("Using input size: %sx%s", sample.input_width, sample.input_height)
        if self._should_use_atm_track() and not self._use_autoregressive_history_template_mode():
            self.logger.info("Using chunk-wise ATM-rendered track maps; each inference window recomputes ATM tracks.")
            self.logger.info("Inference track rendering noise disabled.")
        elif sample.track_video is not None:
            self.logger.info("Using precomputed track maps: %s", tuple(sample.track_video.shape))
            if sample.total_frames > int(sample.track_video.shape[2]):
                self.logger.info(
                    "Episode length exceeds available track horizon (%s > %s); later chunks will reuse the final track frame.",
                    sample.total_frames,
                    int(sample.track_video.shape[2]),
                )

    def _predict_video(self, sample: InferenceSample) -> torch.Tensor:
        if self._use_autoregressive_history_template_mode():
            return self._run_autoregressive_history_template_inference(sample)
        if sample.total_frames > int(self.config.num_frames):
            return self._run_chunked_inference(sample)
        return self._run_single_pass_inference(sample)

    def _run_single_pass_inference(self, sample: InferenceSample) -> torch.Tensor:
        history_frames = int(self.config.num_history_frames)
        action = self._trim_action(sample.action, sample.total_frames)
        infer_frames = self._align_num_frames(sample.total_frames)
        infer_action = self._slice_and_pad_action(
            action=action,
            start=0,
            target_frames=sample.total_frames,
            infer_frames=infer_frames,
        )
        if self._should_use_atm_track():
            infer_track = self._predict_track_video_for_window(
                sample=sample.sample_index,
                metadata_entry=sample.metadata_entry,
                input_video=sample.original_video[:, :, :history_frames],
                task_emb=sample.atm_task_emb,
                chunk_start=0,
                target_frames=sample.total_frames,
                infer_frames=infer_frames,
                num_views=sample.num_views,
                height=sample.input_height,
                width=sample.input_width,
            )
        else:
            infer_track = self._slice_and_pad_track_video(
                track_video=self._trim_track_video(sample.track_video, sample.total_frames),
                start=0,
                target_frames=sample.total_frames,
                infer_frames=infer_frames,
            )
        self._maybe_export_diagnostic_track_comparison(
            sample=sample,
            chunk_start=0,
            chunk_track=infer_track,
            target_frames=sample.total_frames,
            infer_frames=infer_frames,
        )
        self._log_diagnostic_chunk_state(
            sample=sample,
            chunk_start=0,
            input_video=sample.original_video[:, :, :history_frames],
            action=infer_action,
            track=infer_track,
        )
        predicted_video = self._call_pipeline(
            sample,
            input_video=sample.original_video[:, :, :history_frames],
            action=infer_action,
            track=infer_track,
            infer_frames=infer_frames,
        )
        predicted_video = predicted_video.detach().cpu()[:, :, :sample.total_frames]
        self._log_diagnostic_chunk_state(
            sample=sample,
            chunk_start=0,
            input_video=sample.original_video[:, :, :history_frames],
            action=infer_action,
            track=infer_track,
            predicted_video=predicted_video,
        )
        return self._restore_history_frames(predicted_video, sample.original_video, history_frames)

    def _run_autoregressive_history_template_inference(self, sample: InferenceSample) -> torch.Tensor:
        history_frames = int(self.config.num_history_frames)
        chunk_size = int(self.config.num_frames)
        future_frames = chunk_size - history_frames
        if future_frames <= 0:
            raise ValueError(
                f"Autoregressive history-template inference requires num_frames > num_history_frames, got {chunk_size} and {history_frames}"
            )
        if sample.total_frames < history_frames:
            raise ValueError(
                f"Sample has {sample.total_frames} frames, smaller than num_history_frames={history_frames}"
            )

        action = self._trim_action(sample.action, sample.total_frames)
        track_video = self._trim_track_video(sample.track_video, sample.total_frames)
        generated_frames: List[torch.Tensor] = [
            sample.original_video[:, :, frame_idx].clone()
            for frame_idx in range(history_frames)
        ]

        chunk_idx = 0
        while len(generated_frames) < sample.total_frames:
            future_start = len(generated_frames)
            remaining_future = sample.total_frames - future_start
            current_future = min(future_frames, remaining_future)
            requested_frames = history_frames + current_future
            infer_frames = self._align_num_frames(requested_frames)

            history_indices = self._build_autoregressive_history_indices(
                len(generated_frames),
                history_frames,
            )
            chunk_input_video = torch.stack(
                [generated_frames[index] for index in history_indices],
                dim=2,
            )
            chunk_action = self._build_autoregressive_action_condition(
                action=action,
                history_indices=history_indices,
                future_start=future_start,
                future_count=current_future,
                infer_frames=infer_frames,
            )
            chunk_track = self._build_autoregressive_track_condition(
                track_video=track_video,
                history_indices=history_indices,
                future_start=future_start,
                future_count=current_future,
                infer_frames=infer_frames,
            )
            chunk_seed = None if self.config.seed is None else int(self.config.seed) + chunk_idx
            chunk_video = self._call_pipeline(
                sample,
                input_video=chunk_input_video,
                action=chunk_action,
                track=chunk_track,
                infer_frames=infer_frames,
                seed=chunk_seed,
                use_history_condition_noise_in_inference=True,
            )
            chunk_video = chunk_video.detach().cpu()[:, :, :infer_frames]
            future_video = chunk_video[:, :, history_frames:]
            append_count = min(current_future, int(future_video.shape[2]))
            if append_count <= 0:
                raise RuntimeError(
                    "Autoregressive history-template inference produced no future frames for the current chunk."
                )

            for frame_idx in range(append_count):
                generated_frames.append(future_video[:, :, frame_idx].clone())

            if self._is_main_process():
                self.logger.info(
                    "Autoregressive rollout progress: %s/%s frames",
                    len(generated_frames),
                    sample.total_frames,
                )
            chunk_idx += 1

        return torch.stack(generated_frames, dim=2)[:, :, :sample.total_frames]

    def _run_chunked_inference(self, sample: InferenceSample) -> torch.Tensor:
        history_frames = int(self.config.num_history_frames)
        chunk_size = int(self.config.num_frames)
        chunk_stride = chunk_size - history_frames
        if chunk_stride <= 0:
            raise ValueError(
                f"Chunked inference requires num_frames > num_history_frames, got {chunk_size} and {history_frames}"
            )

        action = self._trim_action(sample.action, sample.total_frames)
        track_video = self._trim_track_video(sample.track_video, sample.total_frames)
        chunk_starts = [0]
        while chunk_starts[-1] + chunk_stride + chunk_size <= sample.total_frames:
            chunk_starts.append(chunk_starts[-1] + chunk_stride)

        predicted_chunks: List[torch.Tensor] = []
        last_input_video = sample.original_video[:, :, :history_frames]
        for chunk_idx, start in enumerate(chunk_starts):
            chunk_frames = min(chunk_size, sample.total_frames - start)
            infer_frames = self._align_num_frames(chunk_frames)
            chunk_action = self._slice_and_pad_action(
                action=action,
                start=start,
                target_frames=chunk_frames,
                infer_frames=infer_frames,
            )
            chunk_input_video = last_input_video
            if self._should_use_atm_track():
                chunk_track = self._predict_track_video_for_window(
                    sample=sample.sample_index,
                    metadata_entry=sample.metadata_entry,
                    input_video=chunk_input_video,
                    task_emb=sample.atm_task_emb,
                    chunk_start=start,
                    target_frames=chunk_frames,
                    infer_frames=infer_frames,
                    num_views=sample.num_views,
                    height=sample.input_height,
                    width=sample.input_width,
                )
            else:
                chunk_track = self._slice_and_pad_track_video(
                    track_video=track_video,
                    start=start,
                    target_frames=chunk_frames,
                    infer_frames=infer_frames,
                )
            self._maybe_export_diagnostic_track_comparison(
                sample=sample,
                chunk_start=start,
                chunk_track=chunk_track,
                target_frames=chunk_frames,
                infer_frames=infer_frames,
            )
            self._log_diagnostic_chunk_state(
                sample=sample,
                chunk_start=start,
                input_video=chunk_input_video,
                action=chunk_action,
                track=chunk_track,
            )
            chunk_video = self._call_pipeline(
                sample,
                input_video=chunk_input_video,
                action=chunk_action,
                track=chunk_track,
                infer_frames=infer_frames,
            )
            chunk_video = chunk_video.detach().cpu()[:, :, :chunk_frames]
            self._log_diagnostic_chunk_state(
                sample=sample,
                chunk_start=start,
                input_video=chunk_input_video,
                action=chunk_action,
                track=chunk_track,
                predicted_video=chunk_video,
            )
            chunk_video = self._restore_history_frames(chunk_video, chunk_input_video, history_frames)
            if int(chunk_video.shape[2]) > 0:
                last_input_video = chunk_video[:, :, -history_frames:].contiguous()
            if chunk_idx > 0:
                chunk_video = chunk_video[:, :, history_frames:]
            if int(chunk_video.shape[2]) > 0:
                predicted_chunks.append(chunk_video)

        if predicted_chunks:
            return torch.cat(predicted_chunks, dim=2)
        return sample.original_video[:, :, :0]

    def _call_pipeline(
        self,
        sample: InferenceSample,
        *,
        input_video: torch.Tensor,
        action,
        track,
        infer_frames: int,
        seed=None,
        use_history_condition_noise_in_inference: bool = False,
    ) -> torch.Tensor:
        predicted_video = self.pipeline(
            prompt=sample.prompt,
            negative_prompt=self.config.negative_prompt,
            prompt_emb=sample.prompt_emb,
            negative_prompt_emb=sample.negative_prompt_emb,
            input_video=input_video,
            action=action,
            track=track,
            track_context_scale=float(getattr(self.config, "track_context_scale", 1.0)),
            seed=self.config.seed if seed is None else seed,
            tiled=False,
            height=sample.input_height,
            width=sample.input_width,
            num_frames=infer_frames,
            num_history_frames=int(self.config.num_history_frames),
            cfg_scale=self.config.cfg_scale,
            num_inference_steps=self.config.num_inference_steps,
            use_history_condition_noise_in_inference=use_history_condition_noise_in_inference,
            progress_bar_cmd=self._progress_bar_cmd(),
        )
        if not isinstance(predicted_video, torch.Tensor) or predicted_video.ndim != 5:
            raise TypeError(f"Pipeline output must be (V,C,T,H,W), got {type(predicted_video)}")
        return predicted_video

    def _finalize_predicted_video(
        self,
        predicted_video: torch.Tensor,
        original_video: torch.Tensor,
    ) -> torch.Tensor:
        expected_frames = int(original_video.shape[2])
        pred_frames = int(predicted_video.shape[2])
        if pred_frames > expected_frames:
            return predicted_video[:, :, :expected_frames]
        if pred_frames < expected_frames:
            self.logger.warning("Generated %s frames, expected %s", pred_frames, expected_frames)
        return predicted_video

    def _record_generated_sample(
        self,
        video_path: Path,
        prompt,
        episode_index: int,
        sample_index: int,
    ) -> None:
        if prompt is None:
            raise ValueError("`prompt` must be a non-empty string for standalone metric evaluation.")
        self.generated_sample_records.append(
            {
                "video_path": str(video_path.resolve()),
                "prompt": str(prompt),
                "episode_index": int(episode_index),
                "sample_index": int(sample_index),
                "rank": int(self.dist_context.rank),
            }
        )

    def _finalize_sample_records(self, output_dir: Path) -> List[Dict]:
        merged_records = self._gather_sample_records()
        if not self.dist_context.is_main_process:
            return []
        merged_records.sort(
            key=lambda record: (
                int(record.get("sample_index", -1)),
                int(record.get("episode_index", -1)),
                str(record.get("video_path", "")),
            )
        )
        records_path = merged_sample_records_path(output_dir)
        write_jsonl_records(records_path, merged_records)
        self.logger.info("Saved sample records: %s", records_path.resolve())
        return merged_records

    def _gather_sample_records(self) -> List[Dict]:
        local_records = list(self.generated_sample_records)
        if not self.dist_context.enabled:
            return local_records
        gathered_records: List[Optional[List[Dict]]] = [None for _ in range(self.dist_context.world_size)]
        dist.all_gather_object(gathered_records, local_records)
        if not self.dist_context.is_main_process:
            return []
        merged_records: List[Dict] = []
        for rank_records in gathered_records:
            if rank_records:
                merged_records.extend(rank_records)
        return merged_records

    def _is_main_process(self) -> bool:
        return self.dist_context.is_main_process

    def _progress_bar_cmd(self):
        return tqdm if self._is_main_process() else self._noop_progress_bar

    @staticmethod
    def _noop_progress_bar(iterable, *args, **kwargs):
        return iterable

    @staticmethod
    def _align_num_frames(num_frames: int) -> int:
        if (num_frames - 1) % 4 == 0:
            return num_frames
        return num_frames + (4 - ((num_frames - 1) % 4))

    @staticmethod
    def _trim_track_video(track_video: Optional[torch.Tensor], total_frames: int) -> Optional[torch.Tensor]:
        if track_video is None:
            return None
        return track_video[:, :, :total_frames]

    @staticmethod
    def _slice_and_pad_action(action, start: int, target_frames: int, infer_frames: int):
        if action is None:
            return None
        chunk_action = action[:, start : start + target_frames]
        current_frames = int(chunk_action.shape[1])
        if current_frames >= infer_frames:
            return chunk_action[:, :infer_frames]
        if current_frames <= 0:
            return chunk_action
        pad_frames = infer_frames - current_frames
        if isinstance(chunk_action, torch.Tensor):
            pad = chunk_action[:, -1:, :].repeat(1, pad_frames, 1)
            return torch.cat([chunk_action, pad], dim=1)
        pad = np.repeat(chunk_action[:, -1:, :], repeats=pad_frames, axis=1)
        return np.concatenate([chunk_action, pad], axis=1)

    @staticmethod
    def _slice_and_pad_track_video(
        track_video: Optional[torch.Tensor],
        start: int,
        target_frames: int,
        infer_frames: int,
    ) -> Optional[torch.Tensor]:
        if track_video is None:
            return None
        chunk_track = track_video[:, :, start : start + target_frames]
        current_frames = int(chunk_track.shape[2])
        if current_frames >= infer_frames:
            return chunk_track[:, :, :infer_frames]
        if current_frames <= 0:
            if int(track_video.shape[2]) <= 0:
                return chunk_track
            return track_video[:, :, -1:, :, :].repeat(1, 1, infer_frames, 1, 1)
        pad_frames = infer_frames - current_frames
        pad = chunk_track[:, :, -1:, :, :].repeat(1, 1, pad_frames, 1, 1)
        return torch.cat([chunk_track, pad], dim=2)

    @staticmethod
    def _restore_history_frames(
        predicted_video: torch.Tensor,
        history_source: torch.Tensor,
        history_frames: int,
    ) -> torch.Tensor:
        history_to_copy = min(
            int(history_frames),
            int(predicted_video.shape[2]),
            int(history_source.shape[2]),
        )
        if history_to_copy > 0:
            predicted_video[:, :, :history_to_copy] = history_source[:, :, :history_to_copy]
        return predicted_video

    @staticmethod
    def _build_autoregressive_history_indices(num_generated_frames: int, history_frames: int) -> List[int]:
        if history_frames <= 0:
            return []
        if num_generated_frames < history_frames:
            raise ValueError(
                f"Need at least {history_frames} generated frames, got {num_generated_frames}"
            )
        if history_frames == 1:
            return [num_generated_frames - 1]
        return [0] + list(range(num_generated_frames - (history_frames - 1), num_generated_frames))

    @staticmethod
    def _build_autoregressive_action_condition(
        action,
        *,
        history_indices: List[int],
        future_start: int,
        future_count: int,
        infer_frames: int,
    ):
        if action is None:
            return None

        history_action = action[:, history_indices, :]
        future_action = action[:, future_start : future_start + future_count, :]
        if isinstance(action, torch.Tensor):
            action_cond = torch.cat([history_action, future_action], dim=1)
            current_frames = int(action_cond.shape[1])
            if current_frames < infer_frames:
                pad = action_cond[:, -1:, :].repeat(1, infer_frames - current_frames, 1)
                action_cond = torch.cat([action_cond, pad], dim=1)
            return action_cond

        action_cond = np.concatenate([history_action, future_action], axis=1)
        current_frames = int(action_cond.shape[1])
        if current_frames < infer_frames:
            pad = np.repeat(action_cond[:, -1:, :], repeats=infer_frames - current_frames, axis=1)
            action_cond = np.concatenate([action_cond, pad], axis=1)
        return action_cond

    @staticmethod
    def _build_autoregressive_track_condition(
        track_video: Optional[torch.Tensor],
        *,
        history_indices: List[int],
        future_start: int,
        future_count: int,
        infer_frames: int,
    ) -> Optional[torch.Tensor]:
        if track_video is None:
            return None
        if int(track_video.shape[2]) <= 0:
            return track_video

        max_track_index = int(track_video.shape[2]) - 1
        clamped_history_indices = [min(max(index, 0), max_track_index) for index in history_indices]
        history_track = track_video[:, :, clamped_history_indices, :, :]
        future_track = InferenceEngine._slice_and_pad_track_video(
            track_video=track_video,
            start=future_start,
            target_frames=future_count,
            infer_frames=future_count,
        )
        if future_track is None:
            return history_track
        track_cond = torch.cat([history_track, future_track], dim=2)
        current_frames = int(track_cond.shape[2])
        if current_frames < infer_frames:
            pad = track_cond[:, :, -1:, :, :].repeat(1, 1, infer_frames - current_frames, 1, 1)
            track_cond = torch.cat([track_cond, pad], dim=2)
        return track_cond

    @staticmethod
    def _trim_action(action, total_frames: int):
        if action is None:
            return None
        return action[:, :total_frames]
