def debug_on():
    import sys, os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.argv = [
        "examples/wanvideo/model_training/train.py",
        "--dataset_base_path", "/data_jbx/Codes/Diffsynth_Wan2.2_plus/data/4_4_four_tasks_wan",
        "--dataset_metadata_path", "/data_jbx/Codes/Diffsynth_Wan2.2_plus/data/4_4_four_tasks_wan/meta/episodes_train.track_bert.jsonl",
        "--action_stat_path", "/data_jbx/Codes/Diffsynth_Wan2.2_plus/data/4_4_four_tasks_wan/meta/stat.json",
        "--action_type", "action_pose",
        "--height", "480",
        "--width", "640",
        "--dataset_num_workers", "4",
        "--num_frames", "17",
        "--spatial_division_factor", "32",
        "--dataset_repeat", "1",
        "--model_paths", "/data1/modelscope/models/Wan-AI/Wan2.2-TI2V-5B",
        "--ckpt_path", "/data_jbx/Codes/Diffsynth_Wan2.2_plus/Ckpt/4_6_robot_four_task/epoch-199/epoch-199.safetensors",
        "--trainable_models", "track_context",
        "--learning_rate", "8e-5",
        "--num_epochs", "200",
        "--mixed_precision", "bf16",
        "--output_path", "Ckpt/tmp",
        "--gradient_accumulation_steps", "8",
        "--use_swanlab", "0",
        "--swanlab_experiment_name", "Wan_ATM_Map_0413",
        "--load_modules", "dit,text:emb,vae,image:off,action:noise,trackctx",
        "--num_history_frames", "1",
        "--history_template_sampling", "0",
        "--track_apply_noise", "1",
        "--track_noise_corrupt_ratio", "0.3",
        "--track_noise_offset_scale", "0.008",
        "--track_noise_drift_scale", "0.002",
        "--track_noise_dropout_ratio", "0.1",
        "--track_noise_warmup_frames", "3",
        "--track_context_scale", "0.1",
        "--enable_dit_lora", "0",
        "--lora_target_modules", "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank", "32",
    ]
# debug_on()

import torch, os, argparse, accelerate, random, json
import numpy as np
from diffsynth.core import load_wan_checkpoint_into_pipeline
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.pipelines.wan_video_data import build_wan_video_dataset
from diffsynth.pipelines.wan_video_spec import WanModuleSpec
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_DIT_LORA_TARGET_MODULES = "q,k,v,o,ffn.0,ffn.2"
DEFAULT_DIT_LORA_RANK = 32

def set_global_seed(seed: int = 42) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_csv_arg(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_dit_lora_args(args):
    lora_base_model = args.lora_base_model.strip() if isinstance(args.lora_base_model, str) else args.lora_base_model
    if lora_base_model == "":
        lora_base_model = None
    if not args.lora_checkpoint:
        args.lora_checkpoint = None

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


def resolve_trainable_models(trainable_models, action_enabled, track_context_enabled, enable_dit_lora=False):
    models = split_csv_arg(trainable_models)
    if len(models) == 0:
        models = [] if enable_dit_lora else ["dit"]
    if enable_dit_lora:
        models = [model_name for model_name in models if model_name != "dit"]
    if action_enabled and len(models) > 0 and "action_encoder" not in models and all(model_name == "dit" for model_name in models):
        models.append("action_encoder")
    if track_context_enabled and "track_context" not in models:
        models.append("track_context")
    return ",".join(models) if len(models) > 0 else None


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        modules=("dit", "text", "vae", "image", "action"),
        fp8_models=None,
        offload_models=None,
        ckpt_path=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_history_frames=1,
        history_template_sampling=0,
        num_frames=17,
        track_context_scale=1.0,
    ):
        super().__init__()
        module_spec = WanModuleSpec.parse(modules)
        module_list = list(module_spec.modules)
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(tokenizer_path) if module_spec.enable_text_encoder and tokenizer_path else None
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            modules=module_list
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        if ckpt_path is not None:
            load_wan_checkpoint_into_pipeline(
                self.pipe,
                ckpt_path,
                torch_dtype=self.pipe.torch_dtype,
                device="cpu",
                message_prefix="Loading training weights from checkpoint",
            )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.num_history_frames = num_history_frames
        self.history_template_sampling = int(history_template_sampling)
        self.track_context_scale = float(track_context_scale)

    def get_pipeline_inputs(self, data): 
        inputs_posi = {
            "prompt": data.get("prompt"),
            "prompt_emb": data.get("prompt_emb"),
        }
        inputs_nega = {
            "negative_prompt": data.get("negative_prompt"),
            "prompt_emb": data.get("negative_prompt_emb"),
        }
        inputs_shared = {
            "input_video": data["video"],
            "action": data.get("action"),
            "track": data.get("track"),
            "height": int(data["video"].shape[-2]),
            "width": int(data["video"].shape[-1]),
            "num_frames": int(data["video"].shape[2]),
            "num_history_frames": self.num_history_frames,
            "history_template_sampling": self.history_template_sampling,
            "temporal_future_start": data.get("temporal_future_start"),
            "track_context_scale": self.track_context_scale,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][:, :, 0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser = add_action_config(parser)
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    set_global_seed(args.seed)
    args = normalize_dit_lora_args(args)
    data_file_keys = [key.strip() for key in args.data_file_keys.split(",") if key.strip()]
    module_spec = WanModuleSpec.parse(args.load_modules)
    runtime = module_spec.build_runtime(args.model_paths, data_file_keys)
    modules = runtime.modules
    data_file_keys = runtime.data_file_keys
    action_enabled = runtime.action_enabled
    track_context_enabled = runtime.track_context_enabled

    trainable_models = resolve_trainable_models(
        args.trainable_models,
        action_enabled=action_enabled,
        track_context_enabled=track_context_enabled,
        enable_dit_lora=bool(args.enable_dit_lora),
    )
    args.trainable_models = trainable_models
    model_paths_json = json.dumps(runtime.model_paths)
    tokenizer_path = runtime.tokenizer_path
    log_with = []
    if getattr(args, "use_wandb", False):
        log_with.append("wandb")
    if getattr(args, "use_swanlab", False):
        log_with.append("swanlab")
    log_with = log_with if len(log_with) > 0 else None
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_wan_video_dataset(
        runtime,
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_history_frames=args.num_history_frames,
        repeat=args.dataset_repeat,
        resize_mode=args.resize_mode,
        max_pixels=args.max_pixels,
        data_file_keys=data_file_keys,
        action_type=args.action_type,
        action_stat_path=args.action_stat_path,
        history_template_sampling=args.history_template_sampling,
        height_division_factor=args.spatial_division_factor,
        width_division_factor=args.spatial_division_factor,
        time_division_factor=4,
        time_division_remainder=1,
        track_num_points=args.track_num_points,
        track_point_radius=args.track_point_radius,
        track_seed=args.track_seed,
        track_apply_noise=bool(args.track_apply_noise),
        track_noise_corrupt_ratio=args.track_noise_corrupt_ratio,
        track_noise_offset_scale=args.track_noise_offset_scale,
        track_noise_drift_scale=args.track_noise_drift_scale,
        track_noise_dropout_ratio=args.track_noise_dropout_ratio,
        track_noise_warmup_frames=args.track_noise_warmup_frames,
    )
    model = WanTrainingModule(
        model_paths=model_paths_json,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=tokenizer_path,
        trainable_models=trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        modules=modules,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        ckpt_path=args.ckpt_path,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_history_frames=args.num_history_frames,
        history_template_sampling=args.history_template_sampling,
        num_frames=args.num_frames,
        track_context_scale=args.track_context_scale,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        config=build_grouped_config(parser, args),
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
