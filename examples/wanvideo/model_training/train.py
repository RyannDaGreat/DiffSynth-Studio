import torch, os, json
import sys
sys.path.append("/root/CleanCode/Github/DiffSynth-Studio")
from ryan_utils import debug_print, enable_line_tracing, set_debug_print_ranks
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        debug_print("WanTrainingModule.__init__: begin")
        debug_print(f"WanTrainingModule.__init__: parse_model_configs(model_paths set={model_paths is not None}, model_id_with_origin_paths set={model_id_with_origin_paths is not None})")
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        debug_print(f"WanTrainingModule.__init__: parsed {len(model_configs)} model_configs")
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs)
        debug_print("WanTrainingModule.__init__: pipeline created")

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        debug_print("WanTrainingModule.__init__: switch_pipe_to_training_mode done")
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        debug_print(f"WanTrainingModule.__init__: extra_inputs={self.extra_inputs}, timestep_boundary=({self.min_timestep_boundary}, {self.max_timestep_boundary})")
        
        
    def forward_preprocess(self, data):
        debug_print("WanTrainingModule.forward_preprocess: start")
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            debug_print(f"WanTrainingModule.forward_preprocess: running unit {unit.__class__.__name__}")
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        debug_print("WanTrainingModule.forward_preprocess: done")
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None:
            debug_print("WanTrainingModule.forward: preprocessing inputs")
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        debug_print("WanTrainingModule.forward: computing training_loss")
        loss = self.pipe.training_loss(**models, **inputs)
        debug_print(f"WanTrainingModule.forward: loss computed -> {float(loss.detach().cpu()):.6f}")
        return loss


if __name__ == "__main__":
    debug_print("train.py main: building parser")
    parser = wan_parser()
    args = parser.parse_args()
    set_debug_print_ranks(args.debug_print_ranks)
    if getattr(args, "debug_print_line_tracing", False):
        # WARNING: extremely noisy. Set include_libs=False to limit to repo files only.
        enable_line_tracing(include_libs=True)
        debug_print("train.py main: line tracing enabled via --debug_print_line_tracing")

    # Set global flag for random weights if requested
    if getattr(args, "skip_model_loading", False):
        debug_print("train.py main: setting SKIP_MODEL_LOADING=True for fast dataloader testing")
        from diffsynth.models.utils import SKIP_MODEL_LOADING
        import diffsynth.models.utils as utils
        utils.SKIP_MODEL_LOADING = True
    else:
        debug_print("train.py main: SKIP_MODEL_LOADING=False, using normal weight loading")

    debug_print(f"train.py main: args parsed; dataset_base_path={args.dataset_base_path}, metadata={args.dataset_metadata_path}")
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
    )
    debug_print(f"train.py main: dataset built; len={len(dataset)} load_from_cache={dataset.load_from_cache}")
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    debug_print("train.py main: model created")
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    debug_print(f"train.py main: launching training -> output_path={args.output_path}")
    launch_training_task(dataset, model, model_logger, args=args)
    debug_print("train.py main: training finished")
