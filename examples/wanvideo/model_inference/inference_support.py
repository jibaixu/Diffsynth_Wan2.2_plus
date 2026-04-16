import json
import gc
import logging
import os
from datetime import timedelta
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from peft import LoraConfig, inject_adapter_in_model

from diffsynth.core import load_state_dict
from diffsynth.core.loader.file import load_keys_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.pipelines.wan_video_spec import WanModuleSpec, WanRuntimeConfig
from diffsynth.utils.data import save_video


_TORCHRUN_ENV_KEYS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
_ACTION_PREFIX = "pipe.action_encoder."
_TRACK_CONTEXT_PREFIX = "pipe.track_context."
_UNSUPPORTED_PREFIXES = ("action_encoder.", "dit.")
_DIT_LORA_PREFIXES = ("pipe.dit.", "dit.")
_LORA_STATE_KEY_SUFFIXES = (
    "lora_A.weight",
    "lora_B.weight",
    "lora_A.default.weight",
    "lora_B.default.weight",
)
DEFAULT_DIT_LORA_TARGET_MODULES = ("q", "k", "v", "o", "ffn.0", "ffn.2")
DEFAULT_DIT_LORA_RANK = 32


@dataclass(frozen=True)
class CheckpointOverlaySummary:
    dit_key_count: int = 0
    dit_lora_key_count: int = 0
    action_key_count: int = 0
    track_context_key_count: int = 0

    @property
    def is_track_context_only(self) -> bool:
        return (
            self.track_context_key_count > 0
            and self.dit_key_count == 0
            and self.dit_lora_key_count == 0
            and self.action_key_count == 0
        )


def resolve_optional_path(path_value, base_dir: str):
    if path_value in (None, ""):
        return None
    if isinstance(path_value, os.PathLike):
        path_value = os.fspath(path_value)
    if isinstance(path_value, str) and os.path.isabs(path_value):
        return path_value
    if isinstance(path_value, str):
        return os.path.join(base_dir, path_value)
    return path_value


def flatten_grouped_config(grouped_config: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in grouped_config.values():
        if isinstance(value, dict):
            merged.update(value)
    return merged


def load_checkpoint_grouped_config(ckpt_path: Optional[str]) -> Dict[str, Any]:
    if not ckpt_path:
        return {}
    path = Path(ckpt_path)
    config_path = path / "config.json" if path.is_dir() else path.parent / "config.json"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_flat_config_defaults(ckpt_path: Optional[str]) -> Dict[str, Any]:
    grouped_config = load_checkpoint_grouped_config(ckpt_path)
    if not grouped_config:
        return {}
    if any(isinstance(value, dict) for value in grouped_config.values()):
        grouped_config = flatten_grouped_config(grouped_config)
    if "base_ckpt_path" in grouped_config and "base_checkpoint_path" not in grouped_config:
        grouped_config["base_checkpoint_path"] = grouped_config["base_ckpt_path"]
    return grouped_config


def _split_csv_arg(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _is_lora_state_key(key: str) -> bool:
    return key.endswith(_LORA_STATE_KEY_SUFFIXES)


def _normalize_dit_lora_key(key: str) -> str:
    for prefix in _DIT_LORA_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")


def _classify_checkpoint_key(
    key: str,
    dit_key_filter=None,
) -> tuple[str, Optional[str]]:
    if key.startswith(_ACTION_PREFIX):
        return "action_encoder", key[len(_ACTION_PREFIX):]
    if key.startswith(_TRACK_CONTEXT_PREFIX):
        return "track_context", key[len(_TRACK_CONTEXT_PREFIX):]
    if _is_lora_state_key(key):
        return "dit_lora", _normalize_dit_lora_key(key)
    if key.startswith(_UNSUPPORTED_PREFIXES):
        return "ignore", None
    if key.startswith("pipe."):
        return "ignore", None
    if dit_key_filter is not None and not dit_key_filter(key):
        return "skip", None
    return "dit", key


@dataclass(frozen=True)
class DistributedInferenceContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cuda"

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed_inference(logger: Optional[logging.Logger] = None) -> DistributedInferenceContext:
    env_present = [key for key in _TORCHRUN_ENV_KEYS if key in os.environ]
    if env_present and len(env_present) != len(_TORCHRUN_ENV_KEYS):
        missing = [key for key in _TORCHRUN_ENV_KEYS if key not in os.environ]
        raise RuntimeError(f"Incomplete torchrun environment: missing {missing}")

    if not env_present:
        if logger is not None:
            logger.info("torchrun environment not detected; running single-process inference.")
        return DistributedInferenceContext()

    if not torch.cuda.is_available():
        raise RuntimeError("torchrun inference requires CUDA, but CUDA is not available.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in the current PyTorch build.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30),
        )

    context = DistributedInferenceContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=f"cuda:{local_rank}",
    )
    if logger is not None and rank == 0:
        logger.info(
            "Initialized torchrun inference: rank %s/%s on %s",
            context.rank,
            context.world_size,
            context.device,
        )
    return context


def destroy_distributed_inference(context: Optional[DistributedInferenceContext]) -> None:
    if context is None or not context.enabled:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: Optional[DistributedInferenceContext]) -> None:
    if context is None or not context.enabled:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[context.local_rank])


def broadcast_object(context: Optional[DistributedInferenceContext], value: Any, src: int = 0) -> Any:
    if context is None or not context.enabled:
        return value
    object_list = [value if context.rank == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


@contextmanager
def suppress_stdout_if(enabled: bool):
    if not enabled:
        with nullcontext():
            yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        yield


def merged_sample_records_path(output_dir: Path) -> Path:
    return Path(output_dir) / "sample_records.jsonl"


def write_jsonl_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            json.dump(record, file_handle, ensure_ascii=True)
            file_handle.write("\n")


def _normalize_data_file_keys(data_file_keys) -> list[str]:
    if data_file_keys is None:
        return []
    if isinstance(data_file_keys, str):
        return [item.strip() for item in data_file_keys.split(",") if item.strip()]
    return [str(item).strip() for item in data_file_keys if str(item).strip()]


def build_wan_inference_config(
    values: Dict[str, Any],
    *,
    grouped_config: Optional[Dict[str, Any]] = None,
    data_file_keys=None,
) -> "WanInferenceConfig":
    merged_values = dict(values)
    if "base_ckpt_path" in merged_values and "base_checkpoint_path" not in merged_values:
        merged_values["base_checkpoint_path"] = merged_values["base_ckpt_path"]
    runtime = WanModuleSpec.parse(merged_values.get("load_modules")).build_runtime(
        merged_values.get("model_paths"),
        _normalize_data_file_keys(
            merged_values.get("data_file_keys") if data_file_keys is None else data_file_keys
        ),
    )
    merged_values["load_modules"] = ",".join(runtime.modules)
    merged_values["data_file_keys"] = runtime.data_file_keys
    merged_values["modules"] = runtime.modules
    merged_values["module_bases"] = runtime.module_bases
    merged_values["text_mode"] = runtime.text_mode
    merged_values["action_mode"] = runtime.action_mode
    merged_values["image_mode"] = runtime.image_mode
    merged_values["enable_text"] = runtime.enable_text
    merged_values["enable_text_encoder"] = runtime.enable_text_encoder
    merged_values["action_enabled"] = runtime.action_enabled
    merged_values["has_text_input_for_dit"] = runtime.has_text_input_for_dit
    merged_values["model_paths"] = json.dumps(runtime.model_paths)
    merged_values["tokenizer_path"] = runtime.tokenizer_path
    return WanInferenceConfig(
        values=merged_values,
        grouped_config=grouped_config or {},
        runtime=runtime,
    )


class FrameConverter:
    @staticmethod
    def to_uint8(frame) -> np.ndarray:
        if isinstance(frame, Image.Image):
            return np.array(frame)
        if isinstance(frame, torch.Tensor):
            if frame.dim() == 3 and frame.shape[0] in [1, 3, 4]:
                frame = frame.permute(1, 2, 0)
            frame = frame.detach().to(dtype=torch.float32)
            return FrameConverter._normalize_to_uint8(frame.cpu().numpy())
        if isinstance(frame, np.ndarray):
            return FrameConverter._normalize_to_uint8(frame)
        return frame

    @staticmethod
    def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return array
        max_value = float(array.max())
        min_value = float(array.min())
        if min_value >= -1.0 and max_value <= 1.0:
            if min_value < 0.0:
                array = (array + 1.0) * 127.5
            else:
                array = array * 255.0
        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def ensure_rgb(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return np.stack([frame] * 3, axis=-1)
        return frame


class VideoSaver:
    def __init__(self, fps: int = 5, quality: int = 5, show_progress: bool = True) -> None:
        self.fps = fps
        self.quality = quality
        self.show_progress = show_progress
        self.converter = FrameConverter()

    def save_comparison(
        self,
        original_video,
        predicted_video,
        output_dir: Path,
        video_name: str,
        length_mode: str = "truncate_min",
    ) -> Path:
        if not isinstance(original_video, torch.Tensor) or original_video.ndim != 5:
            raise TypeError("`original_video` must be torch.Tensor with shape (V,C,T,H,W).")
        if not isinstance(predicted_video, torch.Tensor) or predicted_video.ndim != 5:
            raise TypeError("`predicted_video` must be torch.Tensor with shape (V,C,T,H,W).")

        original_video = original_video.detach().to(dtype=torch.float32).cpu()
        predicted_video = predicted_video.detach().to(dtype=torch.float32).cpu()
        num_views = min(int(original_video.shape[0]), int(predicted_video.shape[0]))
        gt_frames = int(original_video.shape[2])
        pred_frames = int(predicted_video.shape[2])
        if length_mode == "truncate_min":
            num_frames = min(gt_frames, pred_frames)
        elif length_mode == "pad_to_pred_black_gt":
            num_frames = max(gt_frames, pred_frames)
        else:
            raise ValueError(f"Unknown length_mode={length_mode!r}")

        comparison_frames: List[np.ndarray] = []
        for frame_idx in range(num_frames):
            rows: List[np.ndarray] = []
            for view_idx in range(num_views):
                gt_frame = self._resolve_frame(
                    video=original_video,
                    frame_idx=frame_idx,
                    view_idx=view_idx,
                    fallback_shape=(int(predicted_video.shape[3]), int(predicted_video.shape[4])),
                )
                pred_frame = self._resolve_frame(
                    video=predicted_video,
                    frame_idx=frame_idx,
                    view_idx=view_idx,
                    fallback_shape=(int(original_video.shape[3]), int(original_video.shape[4])),
                )
                rows.append(np.hstack([gt_frame, pred_frame]))
            comparison_frames.append(np.vstack(rows))

        output_path = output_dir / video_name
        save_video(
            np.asarray(comparison_frames),
            str(output_path),
            fps=self.fps,
            quality=self.quality,
            show_progress=self.show_progress,
        )
        return output_path

    def _resolve_frame(
        self,
        *,
        video: torch.Tensor,
        frame_idx: int,
        view_idx: int,
        fallback_shape: tuple[int, int],
    ) -> np.ndarray:
        if frame_idx < int(video.shape[2]):
            frame = self.converter.to_uint8(video[view_idx, :, frame_idx])
            return self.converter.ensure_rgb(frame)
        height, width = fallback_shape
        return np.zeros((height, width, 3), dtype=np.uint8)


@dataclass
class WanInferenceConfig:
    values: Dict[str, Any]
    runtime: WanRuntimeConfig
    grouped_config: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def metadata_path(self) -> str:
        return str(self._resolve_metadata_path())

    @property
    def resolved_data_base(self) -> str:
        return str(self._resolve_data_base())

    @property
    def videos_path(self) -> str:
        return str(self._resolve_data_base() / "videos")

    @property
    def model_paths_list(self) -> list[Union[str, list[str]]]:
        model_paths = self.values.get("model_paths")
        if not model_paths:
            raise ValueError("Model paths are empty; check `--model_paths` and `--load_modules`.")
        if isinstance(model_paths, str):
            model_paths = json.loads(model_paths)
        if not isinstance(model_paths, (list, tuple)):
            raise ValueError("Model paths must be a list/tuple after parsing.")

        normalized_paths: list[Union[str, list[str]]] = []
        for path in model_paths:
            if isinstance(path, (list, tuple)):
                normalized_paths.append([str(shard_path) for shard_path in path])
            else:
                normalized_paths.append(str(path))
        return normalized_paths

    def build_model_configs(self, offload_device: str = "cpu") -> list[ModelConfig]:
        return [ModelConfig(path=path, offload_device=offload_device) for path in self.model_paths_list]

    def build_tokenizer_config(self) -> Optional[ModelConfig]:
        if not self.enable_text_encoder or not self.tokenizer_path:
            return None
        return ModelConfig(path=self.tokenizer_path)

    def _resolve_metadata_path(self) -> Path:
        metadata_value = self.values.get("dataset_metadata_path")
        if not metadata_value:
            raise ValueError("`dataset_metadata_path` is required for inference.")
        metadata_path = Path(metadata_value)
        if metadata_path.is_absolute():
            return metadata_path
        has_sep = os.sep in metadata_value or (os.altsep and os.altsep in metadata_value)
        if has_sep:
            return metadata_path
        return Path(self.dataset_base_path) / "meta" / metadata_value

    def _resolve_data_base(self) -> Path:
        metadata_path = self._resolve_metadata_path()
        if metadata_path.parent.name == "meta":
            return metadata_path.parent.parent
        return Path(self.dataset_base_path)


class CheckpointPipelineManager:
    _GENERATION_MODULE_NAMES = ("dit", "vae", "text_encoder", "image_encoder", "action_encoder")

    def __init__(
        self,
        config: WanInferenceConfig,
        logger: logging.Logger,
        device: str = "cuda",
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.logger = logger
        self.device = device
        self.verbose = verbose
        self.pipeline: Optional[WanVideoPipeline] = None

    def _info(self, message: str, *args) -> None:
        if self.verbose:
            self.logger.info(message, *args)

    def _warning(self, message: str, *args) -> None:
        if self.verbose:
            self.logger.warning(message, *args)

    @staticmethod
    def _is_vram_managed_module(module) -> bool:
        return bool(getattr(module, "vram_management_enabled", False))

    def _iter_generation_modules(self):
        if self.pipeline is None:
            return
        for name in self._GENERATION_MODULE_NAMES:
            module = getattr(self.pipeline, name, None)
            if module is not None:
                yield name, module

    def discover_checkpoints(self) -> List[Path]:
        checkpoint_path = self.config.values.get("checkpoint_path")
        if not checkpoint_path:
            self._info("Mode: PRETRAINED ONLY (no --ckpt_path)")
            return []
        ckpt_path = Path(checkpoint_path)
        self._info("Mode: SINGLE CHECKPOINT")
        self._info("  - %s", ckpt_path.name)
        return [ckpt_path]

    def _resolve_base_checkpoint(self) -> Optional[Path]:
        base_checkpoint_path = self.config.values.get("base_checkpoint_path")
        if not base_checkpoint_path:
            base_checkpoint_path = self.config.values.get("base_ckpt_path")
        if not base_checkpoint_path:
            return None
        return Path(str(base_checkpoint_path))

    def _summarize_checkpoint_overlay(self, checkpoint: Path) -> CheckpointOverlaySummary:
        summary = CheckpointOverlaySummary()
        for key in load_keys_dict(os.fspath(checkpoint)):
            target, _ = _classify_checkpoint_key(key)
            if target == "dit":
                summary = CheckpointOverlaySummary(
                    dit_key_count=summary.dit_key_count + 1,
                    dit_lora_key_count=summary.dit_lora_key_count,
                    action_key_count=summary.action_key_count,
                    track_context_key_count=summary.track_context_key_count,
                )
            elif target == "dit_lora":
                summary = CheckpointOverlaySummary(
                    dit_key_count=summary.dit_key_count,
                    dit_lora_key_count=summary.dit_lora_key_count + 1,
                    action_key_count=summary.action_key_count,
                    track_context_key_count=summary.track_context_key_count,
                )
            elif target == "action_encoder":
                summary = CheckpointOverlaySummary(
                    dit_key_count=summary.dit_key_count,
                    dit_lora_key_count=summary.dit_lora_key_count,
                    action_key_count=summary.action_key_count + 1,
                    track_context_key_count=summary.track_context_key_count,
                )
            elif target == "track_context":
                summary = CheckpointOverlaySummary(
                    dit_key_count=summary.dit_key_count,
                    dit_lora_key_count=summary.dit_lora_key_count,
                    action_key_count=summary.action_key_count,
                    track_context_key_count=summary.track_context_key_count + 1,
                )
        return summary

    def _warn_if_missing_base_checkpoint(self, checkpoints: List[Path]) -> None:
        if self._resolve_base_checkpoint() is not None or len(checkpoints) == 0:
            return
        checkpoint = Path(checkpoints[0])
        checkpoint_summary = self._summarize_checkpoint_overlay(checkpoint)
        if not checkpoint_summary.is_track_context_only:
            return

        checkpoint_defaults = load_flat_config_defaults(os.fspath(checkpoint))
        trainable_models = set(_split_csv_arg(checkpoint_defaults.get("trainable_models")))
        if trainable_models not in (set(), {"track_context"}):
            return

        self._warning(
            "Checkpoint %s contains only track_context weights (%s keys) and no DiT/action/LoRA weights. "
            "If it was trained as an overlay on a robot base checkpoint, pass --base_ckpt_path so inference "
            "loads that base before applying this checkpoint.",
            checkpoint,
            checkpoint_summary.track_context_key_count,
        )

    def initialize_pipeline(self, checkpoints: List[Path]) -> WanVideoPipeline:
        base_checkpoint = self._resolve_base_checkpoint()
        self._warn_if_missing_base_checkpoint(checkpoints)
        if checkpoints:
            first_ckpt = checkpoints[0]
            if base_checkpoint is not None:
                self._info(
                    "Using pretrained WAN weights; will apply base checkpoint %s before checkpoint %s",
                    base_checkpoint,
                    first_ckpt,
                )
            else:
                self._info("Using pretrained WAN weights; will apply checkpoint: %s", first_ckpt)
        elif base_checkpoint is not None:
            self._info("Using pretrained WAN weights; will apply base checkpoint: %s", base_checkpoint)
        else:
            self._info("Using pretrained WAN weights only (no checkpoint overlay).")
        self.pipeline = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=self.config.build_model_configs(offload_device="cpu"),
            tokenizer_config=self.config.build_tokenizer_config(),
            modules=list(self.config.modules)
        )
        self._apply_preset_lora()
        self._configure_dit_lora()
        if base_checkpoint is not None:
            self._load_checkpoint_overlay(base_checkpoint, overlay_name="base checkpoint")
            self._info("")
        self._info("Pipeline initialized successfully!\n")
        return self.pipeline

    def update_checkpoint(self, checkpoint: Optional[Path]) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        if checkpoint is None:
            self._info("No checkpoint provided; skipping overlay update.")
            return
        self._load_checkpoint_overlay(checkpoint)
        self._info("")

    def _lora_enabled(self) -> bool:
        return bool(int(getattr(self.config, "enable_dit_lora", 0)))

    def _get_dit_module(self):
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        dit = getattr(self.pipeline, "dit", None)
        if dit is None:
            raise ValueError("DiT LoRA inference requires `pipeline.dit`, but the current pipeline has no DiT module.")
        return dit

    def _apply_preset_lora(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        preset_lora_path = getattr(self.config, "preset_lora_path", None)
        if not preset_lora_path:
            return
        preset_lora_model = getattr(self.config, "preset_lora_model", None)
        if not preset_lora_model:
            raise ValueError("`--preset_lora_path` requires `--preset_lora_model`.")
        module = getattr(self.pipeline, preset_lora_model, None)
        if module is None:
            raise ValueError(f"`--preset_lora_model {preset_lora_model}` is not available in the inference pipeline.")
        self._info("Fusing preset LoRA into %s from %s", preset_lora_model, preset_lora_path)
        self.pipeline.load_lora(module, preset_lora_path)

    def _configure_dit_lora(self) -> None:
        if self.pipeline is None or not self._lora_enabled():
            return
        lora_base_model = getattr(self.config, "lora_base_model", None)
        if lora_base_model not in (None, "dit"):
            raise ValueError(f"Inference only supports `--lora_base_model dit`, got: {lora_base_model}")

        target_modules = _split_csv_arg(getattr(self.config, "lora_target_modules", None))
        if not target_modules:
            target_modules = list(DEFAULT_DIT_LORA_TARGET_MODULES)
        lora_rank = int(getattr(self.config, "lora_rank", DEFAULT_DIT_LORA_RANK) or DEFAULT_DIT_LORA_RANK)
        lora_target_value = target_modules[0] if len(target_modules) == 1 else target_modules

        dit = self._get_dit_module()
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=lora_target_value)
        dit = inject_adapter_in_model(lora_config, dit)
        torch_dtype = getattr(self.pipeline, "torch_dtype", None)
        if torch_dtype is not None:
            for param in dit.parameters():
                if param.requires_grad:
                    param.data = param.data.to(dtype=torch_dtype)
        setattr(self.pipeline, "dit", dit)
        self._info(
            "Enabled DiT LoRA for inference: target_modules=%s, rank=%s",
            ",".join(target_modules),
            lora_rank,
        )

        lora_checkpoint = getattr(self.config, "lora_checkpoint", None)
        if lora_checkpoint:
            self._load_explicit_lora_checkpoint(Path(lora_checkpoint))

    def _load_component_state(self, name: str, module, state_dict: Dict[str, torch.Tensor]) -> None:
        if not state_dict:
            return
        if module is None:
            self._warning("  - %s weights found (%s keys), but pipeline.%s is None", name, len(state_dict), name)
            return
        load_result = module.load_state_dict(state_dict, strict=False)
        self._info(
            "  - Loaded %s keys: %s (missing=%s, unexpected=%s)",
            name,
            len(state_dict),
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )

    def _load_dit_lora_state(self, state_dict: Dict[str, torch.Tensor], *, source_label: str) -> int:
        if not state_dict:
            return 0
        if not self._lora_enabled():
            self._warning(
                "  - Ignored %s DiT LoRA keys from %s because --enable_dit_lora=0",
                len(state_dict),
                source_label,
            )
            return 0
        dit = self._get_dit_module()
        load_result = dit.load_state_dict(state_dict, strict=False)
        self._info(
            "  - Loaded dit LoRA keys from %s: %s (unexpected=%s)",
            source_label,
            len(state_dict),
            len(load_result.unexpected_keys),
        )
        if len(load_result.unexpected_keys) > 0:
            self._warning(
                "  - DiT LoRA checkpoint %s has %s unexpected keys",
                source_label,
                len(load_result.unexpected_keys),
            )
        return len(state_dict)

    def _load_explicit_lora_checkpoint(self, checkpoint_path: Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        self._info("Loading explicit DiT LoRA checkpoint: %s", checkpoint_path)
        lora_state = load_state_dict(
            os.fspath(checkpoint_path),
            torch_dtype=getattr(self.pipeline, "torch_dtype", None),
            device="cpu",
            key_filter=_is_lora_state_key,
        )
        normalized_lora_state = {
            _normalize_dit_lora_key(key): value
            for key, value in lora_state.items()
            if _is_lora_state_key(key)
        }
        if not normalized_lora_state:
            self._warning("  - No DiT LoRA keys found in explicit checkpoint: %s", checkpoint_path)
            return
        self._load_dit_lora_state(normalized_lora_state, source_label=str(checkpoint_path))

    def _load_checkpoint_overlay(self, checkpoint: Path, *, overlay_name: str = "checkpoint") -> CheckpointOverlaySummary:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        checkpoint = Path(checkpoint)
        checkpoint_path = os.fspath(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self._info("Updating %s weights: %s", overlay_name, checkpoint_path)
        dit = getattr(self.pipeline, "dit", None)
        dit_key_filter = getattr(dit, "should_load_state_dict_key", None)
        dit_key_filter = dit_key_filter if callable(dit_key_filter) else None

        ignored_key_count = sum(
            1
            for key in load_keys_dict(checkpoint_path)
            if _classify_checkpoint_key(key)[0] == "ignore"
        )

        def keep_checkpoint_key(key: str) -> bool:
            target, _ = _classify_checkpoint_key(key, dit_key_filter)
            return target in ("dit", "action_encoder", "track_context", "dit_lora")

        state_dict = load_state_dict(
            checkpoint_path,
            torch_dtype=getattr(self.pipeline, "torch_dtype", None),
            device="cpu",
            key_filter=keep_checkpoint_key,
        )

        dit_state: Dict[str, torch.Tensor] = {}
        dit_lora_state: Dict[str, torch.Tensor] = {}
        action_state: Dict[str, torch.Tensor] = {}
        track_context_state: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            target, normalized_key = _classify_checkpoint_key(key, dit_key_filter)
            if target == "dit":
                dit_state[normalized_key] = value
            elif target == "dit_lora":
                dit_lora_state[normalized_key] = value
            elif target == "action_encoder":
                action_state[normalized_key] = value
            elif target == "track_context":
                track_context_state[normalized_key] = value

        self._load_component_state("dit", getattr(self.pipeline, "dit", None), dit_state)
        self._load_dit_lora_state(dit_lora_state, source_label=checkpoint.name)
        self._load_component_state("action_encoder", getattr(self.pipeline, "action_encoder", None), action_state)
        self._load_component_state("track_context", getattr(self.pipeline, "track_context", None), track_context_state)

        if ignored_key_count > 0:
            self._info("  - Ignored %s keys with unsupported checkpoint prefixes", ignored_key_count)
        return CheckpointOverlaySummary(
            dit_key_count=len(dit_state),
            dit_lora_key_count=len(dit_lora_state),
            action_key_count=len(action_state),
            track_context_key_count=len(track_context_state),
        )

    def prepare_generation_models(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        for _, module in self._iter_generation_modules():
            if self._is_vram_managed_module(module):
                continue
            if isinstance(module, torch.nn.Module):
                module.to(device=self.device)

    def release_generation_models(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        self._info("Releasing WAN generation models before evaluation...")
        if getattr(self.pipeline, "vram_management_enabled", False):
            self.pipeline.load_models_to_device([])
        for _, module in self._iter_generation_modules():
            if self._is_vram_managed_module(module):
                continue
            if isinstance(module, torch.nn.Module):
                module.to(device="cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._info("Released WAN generation models and cleared CUDA cache.")
