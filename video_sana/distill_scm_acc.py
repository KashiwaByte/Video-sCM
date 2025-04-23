# !/bin/python3
# isort: skip_file
import argparse
import math
from operator import truediv
import os
import time
import datetime
from collections import deque
from copy import deepcopy
import torch.nn.utils.clip_grad as clip_grad
from torch.func import jacrev, jvp

import torch
import torch.distributed as dist
import torch._dynamo
import swanlab
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from peft import LoraConfig
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pdb

import sys
import os
# 获取当前脚本所在的文件路径
current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录路径
current_dir = os.path.dirname(current_file_path)
# 获取父目录路径
parent_dir = os.path.dirname(current_dir)
# 将祖父目录添加到 sys.path 中
sys.path.append(parent_dir)

print(parent_dir)

from fastvideo.dataset.latent_datasets import (LatentDataset, latent_collate_function)
from fastvideo.models.mochi_hf.mochi_latents_utils import normalize_dit_input
from fastvideo.utils.checkpoint import (resume_lora_optimizer, save_checkpoint, save_lora_checkpoint)
from fastvideo.utils.communications import (broadcast, sp_parallel_dataloader_wrapper)
from fastvideo.utils.dataset_utils import LengthGroupedSampler
from fastvideo.utils.load import load_transformer
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group, get_sequence_parallel_state,
                                             initialize_sequence_parallel_state)
from fastvideo.utils.validation import log_validation

from diffusion import SCMScheduler
from diffusion.model.respace import compute_density_for_timestep_sampling

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaBlock"


def get_norm(model_pred, norms, gradient_accumulation_steps):
    fro_norm = (
        torch.linalg.matrix_norm(model_pred, ord="fro") /  # codespell:ignore
        gradient_accumulation_steps)
    largest_singular_value = (torch.linalg.matrix_norm(model_pred, ord=2) / gradient_accumulation_steps)
    absolute_mean = torch.mean(torch.abs(model_pred)) / gradient_accumulation_steps
    absolute_max = torch.max(torch.abs(model_pred)) / gradient_accumulation_steps
    dist.all_reduce(fro_norm, op=dist.ReduceOp.AVG)
    dist.all_reduce(largest_singular_value, op=dist.ReduceOp.AVG)
    dist.all_reduce(absolute_mean, op=dist.ReduceOp.AVG)
    norms["fro"] += torch.mean(fro_norm).item()  # codespell:ignore
    norms["largest singular value"] += torch.mean(largest_singular_value).item()
    norms["absolute mean"] += absolute_mean.item()
    norms["absolute max"] += absolute_max.item()


def distill_one_step(
    transformer,
    model_type,
    teacher_transformer,
    optimizer,
    lr_scheduler,
    loader,
    noise_scheduler,
    gradient_accumulation_steps,
    sp_size,
    max_grad_norm,
    uncond_prompt_embed,
    uncond_prompt_mask,
    not_apply_cfg_solver,
    distill_cfg,
    hunyuan_teacher_disable_cfg,
    global_step,
    accelerator,
    args,
):
    total_loss = 0.0
    optimizer.zero_grad()
    model_pred_norm = {
        "fro": 0.0,  # codespell:ignore
        "largest singular value": 0.0,
        "absolute mean": 0.0,
        "absolute max": 0.0,
    }
    for _ in range(gradient_accumulation_steps):
        (
            latents,
            encoder_hidden_states,
            latents_attention_mask,
            encoder_attention_mask,
        ) = next(loader)
        sigma_data=args.sigma_data
        model_input = normalize_dit_input(model_type, latents)
        noise = torch.randn_like(model_input)*sigma_data
        bsz = model_input.shape[0]
        
        # Sample timesteps for SCM training
        if args.weighting_scheme == "logit_normal_trigflow":
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=None,
            )
            timesteps = u.float().to(model_input.device)
        elif args.weighting_scheme == "logit_normal_trigflow_ladd":
            indices = torch.randint(0, len(args.add_noise_timesteps), (bsz,))
            u = torch.tensor([args.add_noise_timesteps[i] for i in indices])
            if len(args.add_noise_timesteps) == 1:
                # zero-SNR
                timesteps = torch.tensor([1.57080 for i in indices]).float().to(model_input.device)
            else:
                timesteps = u.float().to(model_input.device)
        else:
            # Default random sampling
            timesteps = torch.rand(bsz, device=model_input.device) * 1.57080  # pi/2
                
        t = timesteps.view(-1, 1, 1, 1, 1)
        
        # Add noise according to SCM
        z = torch.randn_like(model_input) * sigma_data
        x_t = torch.cos(t) * model_input + torch.sin(t) * z
        
        target_shape = (16, 21, 60, 104)
        patch_size = (1, 2, 2)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                          (patch_size[1] * patch_size[2]) *
                          target_shape[1] / sp_size) * sp_size
        
        # @torch.compile()
        def model_wrapper(scaled_x_t, t):
            pred, logvar = accelerator.unwrap_model(transformer)(
                x=scaled_x_t, context=encoder_hidden_states, t=t.flatten(), seq_len=seq_len, return_logvar=True, jvp=True
            )
            return pred, logvar

        # Predict using transformer
        with torch.autocast("cuda", dtype=torch.bfloat16):

            with torch.no_grad():
                print("教师模型开始计算")
                pretrain_pred = teacher_transformer(
                    x=x_t / sigma_data,
                    context=encoder_hidden_states,
                    t=t.flatten(),
                    seq_len=seq_len,
                    return_logvar=False,
                    jvp=False
                )[0]
                
                dxt_dt = sigma_data * pretrain_pred
                v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
                v_t = torch.cos(t) * torch.sin(t)
                
            with torch.no_grad():    
                print("雅可比矩阵开始计算")
                F_theta_jvp, F_theta_grad = torch.func.jvp(
                    model_wrapper, (x_t / sigma_data, t), (v_x, v_t), has_aux=False
                )

            F_theta, logvar = transformer(
                x=x_t / sigma_data,
                context=encoder_hidden_states,
                t=t.flatten(),
                seq_len=seq_len,
                return_logvar=True,
                jvp=False
            )
            
            logvar = logvar.view(-1, 1, 1, 1, 1)
            F_theta = F_theta[0]
            F_theta = F_theta.clone()
            F_theta_grad = F_theta_grad.view_as(F_theta).detach()
            F_theta_minus = F_theta.detach()
            
            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            
            # Warmup steps
            r = min(1, global_step / args.tangent_warmup_steps)
            second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)
            g = g + second_term
            
            # Tangent normalization
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
            g = g / (g_norm + 0.1)  # 0.1 is the constant c
            
            # Calculate loss with weight and logvar
            sigma = torch.tan(t) * sigma_data
            weight = 1 / sigma
            l2_loss = torch.square(F_theta - F_theta_minus - g)
            
            # Calculate loss with normalization factor
            loss = (weight / torch.exp(logvar)) * l2_loss + logvar
            loss = loss.mean() / gradient_accumulation_steps
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), max_grad_norm)
            
            total_loss += loss.detach().item()
            get_norm(model_pred, model_pred_norm, gradient_accumulation_steps)
    
    optimizer.step()
    lr_scheduler.step()
    
    return total_loss, model_pred_norm


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--output_dir", type=str, default="output")
    # parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    # parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # parser.add_argument("--num_train_epochs", type=int, default=100)
    # parser.add_argument("--save_steps", type=int, default=500)
    # parser.add_argument("--validation_steps", type=int, default=500)
    # parser.add_argument("--logging_steps", type=int, default=10)
    # parser.add_argument("--mixed_precision", type=str, default="bf16")
    # parser.add_argument("--use_lora", action="store_true")
    # parser.add_argument("--lora_r", type=int, default=16)
    # parser.add_argument("--lora_alpha", type=int, default=32)
    # parser.add_argument("--lora_dropout", type=float, default=0.05)
    # parser.add_argument("--lora_bias", type=str, default="none")
    # parser.add_argument("--sigma_data", type=float, default=1.0)
    # parser.add_argument("--logit_mean", type=float, default=0.0)
    # parser.add_argument("--logit_std", type=float, default=1.0)
    # parser.add_argument("--add_noise_timesteps", nargs="+", type=float, default=[1.57080])
    # parser.add_argument("--model_type", type=str, default="wan_scm", help="The type of model to train.")


    parser.add_argument("--model_type", type=str, default="wan_scm", help="The type of model to train.")

    # Scheduler
    parser.add_argument('--predict_flow_v', type=bool, default=True,
                   help='Enable flow velocity prediction')
    parser.add_argument('--noise_schedule', type=str, default='linear_flow',
                    help='Type of noise schedule')
    parser.add_argument('--pred_sigma', type=bool, default=False,
                    help='Enable sigma prediction')
    parser.add_argument('--weighting_scheme', type=str, default='logit_normal_trigflow',
                    help='Weighting scheme for timesteps')
    parser.add_argument('--logit_mean', type=float, default=0.2,
                    help='Mean for logit-normal distribution')
    parser.add_argument('--logit_std', type=float, default=1.6,
                    help='Standard deviation for logit-normal distribution')
    parser.add_argument('--logit_mean_discriminator', type=float, default=-0.6,
                    help='Mean for discriminator logit-normal distribution')
    parser.add_argument('--logit_std_discriminator', type=float, default=1.0,
                    help='Standard deviation for discriminator logit-normal distribution')
    parser.add_argument('--sigma_data', type=float, default=0.5,
                    help='Sigma value for data')
    parser.add_argument('--vis_sampler', type=str, default='scm',
                    help='Visualization sampler type')
    parser.add_argument('--timestep_norm_scale_factor', type=int, default=1000,
                    help='Scale factor for timestep normalization')
    
    
    # sCM
    parser.add_argument("--logvar",type=bool, default=True, help="Whether to use log variance in sCM.")
    parser.add_argument('--class_dropout_prob', type=float, default=0.0,
                   help='Probability of class dropout during training')
    parser.add_argument('--cfg_scale', type=float, default=5.0,
                    help='Configuration scale factor')
    parser.add_argument('--cfg_embed', type=bool, default=True,
                    help='Whether to use configuration embedding')
    parser.add_argument('--cfg_embed_scale', type=float, default=0.1,
                    help='Scale factor for configuration embedding')
    
    # sCM   training arguments
    parser.add_argument('--tangent_warmup_steps', type=int, default=4000,
                   help='Number of warmup steps for tangent learning')
    parser.add_argument('--scm_cfg_scale', type=float, nargs='+', default=[4.0, 4.5, 5.0],
                    help='Configuration scale values for sCM')
    
    
    
    # LADD config
    parser.add_argument('--ladd_multi_scale', type=bool, default=True,
                   help='Enable multi-scale feature for LADD')
    parser.add_argument('--head_block_ids', type=int, nargs='+', default=[2, 8, 14, 19],
                   help='Block IDs for head layers')
    
    # LADD training arguments
    parser.add_argument('--adv_lambda', type=float, default=0.5,
                   help='Weight for adversarial loss')
    parser.add_argument('--scm_lambda', type=float, default=1.0,
                    help='Weight for SCM loss')
    parser.add_argument('--scm_loss', type=bool, default=True,
                    help='Enable SCM loss')
    parser.add_argument('--misaligned_pairs_D', type=bool, default=True,
                    help='Enable misaligned pairs for discriminator')
    parser.add_argument('--discriminator_loss', type=str, default='hinge',
                    help='Type of discriminator loss')
    parser.add_argument('--train_largest_timestep', type=bool, default=True,
                    help='Enable training with largest timestep')
    parser.add_argument('--largest_timestep', type=float, default=1.57080,
                    help='Value of largest timestep')
    parser.add_argument('--largest_timestep_prob', type=float, default=0.5,
                    help='Probability of using largest timestep')
    
    
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str , default= "/run/determined/workdir/data/H800/datasets/webvid-10k/Image-Vid-wan/videos2caption.json")
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t", type=int, default=32, help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str,default=" /run/determined/workdir/data/H800/diffusion/models/Wan2.1-T2V-1.3B")
    parser.add_argument("--dit_model_name_or_path", type=str , default= "/run/determined/workdir/data/H800/diffusion/models/Wan2.1-T2V-1.3B")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--cfg", type=float, default=0.1)

    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str, default="/run/determined/workdir/data/H800/datasets/webvid-10k/Image-Vid-Finetune-wan/validation")
    parser.add_argument("--validation_sampling_steps", type=str, default="25")
    parser.add_argument("--validation_guidance_scale", type=str, default="4.5")

    parser.add_argument("--validation_steps", type=float, default=25)
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/run/determined/workdir/data/H800/datasets/webvid-10k/outputs/wan-1.3B-1e6-16-latent32",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=160,
        help=("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
              " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
              " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=320,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
        default= "True",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
              " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
        default=False,
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha", type=int, default=256, help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank", type=int, default=128, help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
              ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument("--num_euler_timesteps", type=int, default=100)
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--not_apply_cfg_solver",
        action="store_true",
        help="Whether to apply the cfg_solver.",
    )
    parser.add_argument("--distill_cfg", type=float, default=3.0, help="Distillation coefficient.")
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
    parser.add_argument("--scheduler_type", type=str, default="pcm", help="The scheduler type to use.")
    parser.add_argument(
        "--linear_quadratic_threshold",
        type=float,
        default=0.025,
        help="Threshold for linear quadratic scheduler.",
    )
    parser.add_argument(
        "--linear_range",
        type=float,
        default=0.5,
        help="Range for linear quadratic scheduler.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay to apply.")
    parser.add_argument("--multi_phased_distill_schedule", type=str, default="4000-1")
    parser.add_argument("--hunyuan_teacher_disable_cfg", action="store_true")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="bf16",
        help="Weight type to use - fp32 or bf16.",)
    
    args = parser.parse_args()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # Initialize accelerator
    init_process_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
        kwargs_handlers=[init_process_kwargs],
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # Load model and teacher model
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )
    
    teacher_transformer = deepcopy(transformer)
    
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )
        transformer.add_adapter(lora_config)
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_train_epochs,
    )
    
    # Prepare dataset and dataloader
    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    uncond_prompt_embed = train_dataset.uncond_prompt_embed
    uncond_prompt_mask = train_dataset.uncond_prompt_mask
    sampler = (LengthGroupedSampler(
        args.train_batch_size,
        rank=rank,
        world_size=world_size,
        lengths=train_dataset.lengths,
        group_frame=args.group_frame,
        group_resolution=args.group_resolution,
    ) if (args.group_frame or args.group_resolution) else DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=False))
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    
    # Prepare everything with accelerator
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    transformer = transformer.cuda()
    teacher_transformer = teacher_transformer.cuda()
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint is not None:
        if args.use_lora:
            resume_lora_optimizer(args.resume_from_checkpoint, transformer, optimizer)
        else:
            accelerator.load_state(args.resume_from_checkpoint)
    
    loader = sp_parallel_dataloader_wrapper(
    train_dataloader,
    device,
    args.train_batch_size,
    args.sp_size,
    args.train_sp_batch_size,
)
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.num_train_epochs), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            loss, model_pred_norm = distill_one_step(
                transformer=transformer,
                model_type=args.model_type,
                teacher_transformer=teacher_transformer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                loader=loader,
                noise_scheduler=None,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                sp_size=args.sp_size,
                max_grad_norm=args.max_grad_norm,
                uncond_prompt_embed=None,
                uncond_prompt_mask=None,
                not_apply_cfg_solver=True,
                distill_cfg=False,
                hunyuan_teacher_disable_cfg=True,
                global_step=global_step,
                accelerator=accelerator,
                args=args,
            )
            
            train_loss += loss
            
            if global_step % args.logging_steps == 0:
                logs = {
                    "loss": loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                logs.update(model_pred_norm)
                accelerator.log(logs)
            
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    if args.use_lora:
                        save_lora_checkpoint(
                            args.output_dir,
                            transformer,
                            optimizer,
                            global_step,
                        )
                    else:
                        accelerator.save_state()
            
            if global_step % args.validation_steps == 0:
                if accelerator.is_main_process:
                    transformer.eval()
                    log_validation(
                        accelerator,
                        transformer,
                        global_step,
                        args.output_dir,
                    )
                    transformer.train()
            
            global_step += 1
        
        progress_bar.update(1)
        train_loss = train_loss / len(train_dataloader)
        logs = {"epoch": epoch, "train_loss": train_loss}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs)
    
    accelerator.end_training()


if __name__ == "__main__":
    main()