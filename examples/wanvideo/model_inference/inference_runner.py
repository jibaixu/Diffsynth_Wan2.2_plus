"""
机器人视频推理 CLI 入口
负责：命令行参数解析、配置管理、启动推理引擎
"""
import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffsynth.diffusion.parsers import (
    add_action_config,
    add_dataset_base_config,
    add_infer_config,
    add_lora_config,
    add_model_config,
    add_track_context_config,
    add_training_config,
    add_video_size_config,
    build_grouped_config,
)

from inference_support import (
    build_wan_inference_config,
    destroy_distributed_inference,
    initialize_distributed_inference,
    load_flat_config_defaults,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_ATM_CKPT_PATH = (
    "/data_jbx/Codes/ATM/results/track_transformer/"
    "0409_realbot_track_transformer_001B_action_bs_16_grad_acc_4_numtrack_256_ep1001_0047/"
    "model_best.ckpt"
)
DEFAULT_DIT_LORA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
DEFAULT_DIT_LORA_RANK = 32
_EXPLICIT_ONLY_LORA_ARG_NAMES = {
    "enable_dit_lora",
    "lora_base_model",
    "lora_target_modules",
    "lora_rank",
    "lora_checkpoint",
    "preset_lora_path",
    "preset_lora_model",
}


def add_atm_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("atm")
    group.add_argument(
        "--use_atm_track",
        type=int,
        choices=[0, 1],
        default=1,
        help="Enable online ATM track prediction and feed the rendered track maps into WAN trackctx.",
    )
    group.add_argument(
        "--atm_ckpt_path",
        type=str,
        default=DEFAULT_ATM_CKPT_PATH,
        help="Path to the ATM checkpoint used for online track prediction.",
    )
    return parser


def add_diagnostic_config(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("diagnostic")
    group.add_argument(
        "--diagnose_inference",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable extra inference diagnostics, including condition stats and first-chunk track previews.",
    )
    group.add_argument(
        "--diagnostic_max_total_frames",
        type=int,
        default=0,
        help="If >0, truncate each sample to at most this many frames for diagnosis. Useful for first-chunk-only checks.",
    )
    group.add_argument(
        "--diagnostic_export_track_videos",
        type=int,
        choices=[0, 1],
        default=1,
        help="When diagnostics are enabled, export first-chunk ATM-vs-dataset track comparisons.",
    )
    return parser


def normalize_dit_lora_args(args):
    lora_base_model = args.lora_base_model.strip() if isinstance(args.lora_base_model, str) else args.lora_base_model
    if lora_base_model == "":
        lora_base_model = None
    if not args.lora_checkpoint:
        args.lora_checkpoint = None
    if not args.preset_lora_path:
        args.preset_lora_path = None
    if not args.preset_lora_model:
        args.preset_lora_model = None

    if bool(args.enable_dit_lora):
        if lora_base_model not in (None, "dit"):
            raise ValueError(f"--enable_dit_lora only supports LoRA on `dit`, got: {lora_base_model}")
        lora_base_model = "dit"
        if not args.lora_target_modules:
            args.lora_target_modules = DEFAULT_DIT_LORA_TARGET_MODULES
        if args.lora_rank is None:
            args.lora_rank = DEFAULT_DIT_LORA_RANK

    args.lora_base_model = lora_base_model
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Robot inference with WanVideo pipeline")
    parser = add_model_config(parser)
    parser = add_dataset_base_config(parser)
    parser = add_action_config(parser)
    parser = add_track_context_config(parser)
    parser = add_video_size_config(parser)
    parser = add_training_config(parser)
    parser = add_lora_config(parser)
    parser = add_infer_config(parser)
    parser = add_atm_config(parser)
    parser = add_diagnostic_config(parser)

    for action in parser._actions:
        if getattr(action, "dest", None) == "dataset_base_path":
            action.required = False
            break

    pre_args, _ = parser.parse_known_args()
    defaults = load_flat_config_defaults(pre_args.checkpoint_path)
    if defaults:
        known = {action.dest for action in parser._actions if getattr(action, "dest", None)}
        parser.set_defaults(
            **{
                key: value
                for key, value in defaults.items()
                if key in known and key not in _EXPLICIT_ONLY_LORA_ARG_NAMES
            }
        )

    args = parser.parse_args()
    args = normalize_dit_lora_args(args)
    if not args.dataset_base_path:
        raise ValueError("`--dataset_base_path` is required (or provide a config.json next to the checkpoint).")

    grouped_config = build_grouped_config(parser, args) or {}
    return build_wan_inference_config(
        vars(args).copy(),
        grouped_config=grouped_config,
    )


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def main() -> None:
    from inference_engine import InferenceEngine

    config = parse_args()
    logger = setup_logging()
    dist_context = None
    try:
        dist_context = initialize_distributed_inference(logger)
        InferenceEngine(config, logger, dist_context=dist_context).run()
    finally:
        destroy_distributed_inference(dist_context)


if __name__ == "__main__":
    main()
