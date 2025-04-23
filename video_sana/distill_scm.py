# !/bin/python3
# isort: skip_file
import argparse
import math
from operator import truediv
import os
import time
from collections import deque
from copy import deepcopy
import torch.nn.utils.clip_grad as clip_grad
from torch.func import jacrev, jvp

import torch
import torch.distributed as dist
import faulthandler
faulthandler.enable()
import swanlab
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from peft import LoraConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
from fastvideo.utils.fsdp_util import (apply_fsdp_checkpointing, get_dit_fsdp_kwargs)
from fastvideo.utils.load import load_transformer
from fastvideo.utils.parallel_states import (destroy_sequence_parallel_group, get_sequence_parallel_state,
                                             initialize_sequence_parallel_state)
from fastvideo.utils.validation import log_validation

from diffusion import SCMScheduler
from diffusion.model.respace import compute_density_for_timestep_sampling

from models.wan.modules.t5 import T5EncoderModel

from torch.utils.tensorboard import SummaryWriter

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")


def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)


def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


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
    text_encoder,
    writer,

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
        model_input= normalize_dit_input(model_type, latents)
        x0 = model_input * sigma_data
        bsz = x0.shape[0]
        
        # Sample timesteps for SCM training
        if args.weighting_scheme == "logit_normal_trigflow":
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=None,
            )
            timesteps = u.float().to(x0.device)
        elif args.weighting_scheme == "logit_normal_trigflow_ladd":
            indices = torch.randint(0, len(args.add_noise_timesteps), (bsz,))
            u = torch.tensor([args.add_noise_timesteps[i] for i in indices])
            if len(args.add_noise_timesteps) == 1:
                # zero-SNR
                timesteps = torch.tensor([1.57080 for i in indices]).float().to(x0.device)
            else:
                timesteps = u.float().to(x0.device)
        else:
            # Default random sampling
            timesteps = torch.rand(bsz, device=x0.device) * 1.57080  # pi/2
                
        t = timesteps.view(-1, 1, 1, 1, 1)
    
    
        # Add noise according to SCM
        z = torch.randn_like(x0) * sigma_data
        x_t = torch.cos(t) * x0 + torch.sin(t) * z
        # pdb.set_trace()
        
        def model_wrapper(scaled_x_t, t):
            # Get unwrapped model to avoid FSDP issues with jvp
            # unwrapped_transformer = transformer._fsdp_wrapped_module if hasattr(transformer, '_fsdp_wrapped_module') else transformer
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # unwrapped_transformer = unwrapped_transformer.to(device)    
            # pred = unwrapped_transformer(
            #     x=scaled_x_t, context=encoder_hidden_states, t=t.flatten(), seq_len=seq_len, return_logvar=False, jvp=True
            # )
            
            pred = transformer(
                x=scaled_x_t, context=encoder_hidden_states, t=t.flatten(), seq_len=seq_len, return_logvar=False, jvp=True
            )
        
            if isinstance(pred, list):
                pred = tuple(pred)
            return pred

        

        
        target_shape = (16, 21, 60, 104)
        patch_size = (1, 2, 2)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                          (patch_size[1] * patch_size[2]) *
                          target_shape[1] / sp_size) * sp_size
        
        # Predict using transformer
        # pdb.set_trace()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model_kwargs = {
                "x": x_t / sigma_data,
                "context": encoder_hidden_states,
                "t": t.flatten(),
                "seq_len": seq_len,
                "return_logvar": False,
                "jvp": False
            }


            if args.scm_cfg_scale[0] > 1 and args.cfg_embed:    
                # 创建无条件文本嵌入
                # uncond_context = torch.zeros_like(encoder_hidden_states)  # 方法1:全零初始化
                # 或者使用预训练模型的空字符串嵌入
                n_prompt = args.sample_neg_prompt
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                context_null = text_encoder([n_prompt],device)
                context_null = [t.to(device) for t in context_null]     
   
                
                uncond_cfg_model_kwargs = {
                    "x": x_t / sigma_data,
                    "context": context_null, 
                    "t": t.flatten(),
                    "seq_len": seq_len,
                    "return_logvar": False,
                    "jvp": False
                }
                
                with torch.no_grad():
                    print("教师模型CFG预测开始")
                    pretrain_pred_cond = teacher_transformer(**model_kwargs)[0]
                    pretrain_pred_uncond = teacher_transformer(**uncond_cfg_model_kwargs)[0]
                    # pdb.set_trace()

                    # 应用CFG
                    pretrain_pred =  pretrain_pred_uncond  + distill_cfg * (pretrain_pred_cond - pretrain_pred_uncond)
            else:
                with torch.no_grad():
                    print("教师模型开始计算")
                    pretrain_pred = teacher_transformer(**model_kwargs)
                    pretrain_pred = pretrain_pred[0]
            
            print("pretrain_pred mean/std:",pretrain_pred.mean().item(), pretrain_pred.std().item() )
            dxt_dt = sigma_data * pretrain_pred
            v_x = torch.cos(t) * torch.sin(t) * dxt_dt / sigma_data
            v_t = torch.cos(t) * torch.sin(t)
            
            scaled_input = x_t / sigma_data
            print("Scaled input mean/std:", scaled_input.mean().item(), scaled_input.std().item())
            v_x_norm = v_x.norm().item()
            v_t_norm = v_t.norm().item()
            print("Tangent vectors norm:", v_x_norm, v_t_norm)
                
            print("雅可比矩阵开始计算")  
            with torch.no_grad():
                F_theta, F_theta_grad = torch.func.jvp(
                                model_wrapper, (x_t / sigma_data, t), (v_x, v_t), has_aux=False
                            )
                
                print(f"F_theta_grad{F_theta_grad[0].shape}")
                print(f"F_theta{F_theta[0].shape}")
                
            
            
            print("学生模型开始计算")
            F_theta, logvar = transformer(
                x=x_t / sigma_data,
                context=encoder_hidden_states,
                t=t.flatten(),
                seq_len=seq_len,
                return_logvar=True,
                jvp=False
            )
            
            # Clone F_theta to avoid modifying the view
            
            logvar = logvar.view(-1, 1, 1, 1, 1)
            F_theta = F_theta[0]
            F_theta_grad = F_theta_grad[0] 
            F_theta_grad = F_theta_grad.detach()
            F_theta_minus = F_theta.detach()
            
            
            # Warmup steps
            r = min(1, global_step / args.tangent_warmup_steps)
            
            # Calculate gradient g using JVP rearrangement
            g = -torch.cos(t) * torch.cos(t) * (sigma_data * F_theta_minus - dxt_dt)
            second_term = -r * (torch.cos(t) * torch.sin(t) * x_t + sigma_data * F_theta_grad)

        

            
            # pdb.set_trace()
            print(g.shape)
            print(f"F_theta_grad{F_theta_grad.mean()}")
            
            # 计算 g 和 second_term 的数量级（需分离计算图，避免干扰梯度）
            with torch.no_grad():
                # 取绝对值后计算对数，向下取整得到数量级
                g_mean = g.mean()
                second_mean =  second_term.mean()
                g_order = torch.floor(torch.log10(torch.abs(g_mean))).item()
                second_term_order = torch.floor(torch.log10(torch.abs(second_mean))).item()

                # 计算缩放因子（10 的幂次方）
                scale_factor = 10 ** (g_order - second_term_order)
                scale_factor = torch.tensor(scale_factor, dtype=g.dtype, device=g.device)
                print("scale_factor:",scale_factor )
            # 对 second_term 进行缩放（保留梯度）
            second_term_scaled = second_term * scale_factor

            print(f"g:{g.mean()}")
            print(f"g_max:{g.max()}")
            print(f"second:{second_term.mean()}")
            print(f"second—scaled_mean:{second_term_scaled.mean()}")
            print(f"second—scaled_norm:{second_term_scaled.norm().item()}")
            
            
            
            
            g = g+ second_term_scaled
            # g = g + second_term

            # Tangent normalization
            g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3, 4), keepdim=True)
            print(f"g_norm:{g_norm.mean()}")
            print(f"g在维度1的范数: {torch.linalg.vector_norm(g, dim=1).mean()}")
            print(f"g在维度2的范数: {torch.linalg.vector_norm(g, dim=2).mean()}")
            print(f"g在维度3的范数: {torch.linalg.vector_norm(g, dim=3).mean()}")
            print(f"g在维度4的范数: {torch.linalg.vector_norm(g, dim=4).mean()}")

            norm_dim1 = torch.linalg.vector_norm(g, dim=1, keepdim=True)  # 形状 (N, 1, H, W, D)
            norm_dim2 = torch.linalg.vector_norm(g, dim=2, keepdim=True)  # 形状 (N, C, 1, W, D)
            norm_dim3 = torch.linalg.vector_norm(g, dim=3, keepdim=True)  # 形状 (N, C, H, 1, D)
            norm_dim4 = torch.linalg.vector_norm(g, dim=4, keepdim=True)  # 形状 (N, C, H, W, 1)
                        
            g_norm_4 =  (norm_dim1 * norm_dim2 * norm_dim3 * norm_dim4).pow(1/4)
            swanlab.log({"Debug/g":g.mean(),"Debug/g_max":g.max(),"Debug/g_norm":g_norm.mean(),"Debug/g_norm_4":g_norm_4.mean()})

            g = g / (g_norm + 0.1)  # 0.1 is the constant cs
            # g =  g / (g_norm_4 + 0.1) 
            print(f"g_after:{g.mean()}")

            
            # Calculate loss with weight and logvar
            sigma = torch.tan(t) * sigma_data
            weight = 1 / sigma
            l2_loss = torch.square(F_theta - F_theta_minus - g)
            
            # Calculate loss with normalization factor
            loss = (weight / torch.exp(logvar)) * l2_loss + logvar
            # loss = l2_loss 
            # pdb.set_trace()
            loss = loss.mean() 
            
            
            loss_no_logvar = weight * torch.square(F_theta - F_theta_minus - g)
            loss_no_logvar = loss_no_logvar.mean()
            loss_no_weight = l2_loss.mean()
            g_norm = g_norm.mean()
            
        # pdb.set_trace()
        # Calculate model prediction norms
        get_norm(F_theta.detach().float(), model_pred_norm, gradient_accumulation_steps)
        loss.backward()
        
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()

    def compute_grad_norm(parameters):
        # 获取所有梯度的范数（L2 norm）
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
        return total_norm # 返回标量值

   
    # grad_norm = compute_grad_norm(transformer.parameters())
    grad_norm = clip_grad.clip_grad_norm_(transformer.parameters(), max_grad_norm)    
    swanlab.log({"Debug/grad_after":grad_norm.mean() / g_norm_4.mean() })
    params_before = {name: param.data.clone() for name, param in transformer.named_parameters() if param.requires_grad}
    optimizer.step()
    
    # for name, param in transformer.named_parameters():
    #     writer.add_histogram(f'weights/{name}', param, global_step)
    #     writer.add_histogram(f'grads/{name}', param.grad,  global_step)
    # 计算并记录参数更新的幅度
    with torch.no_grad():
        param_changes = {}
        for name, param in transformer.named_parameters():
            if param.requires_grad and name in params_before:
                param_diff = (param.data - params_before[name]).float()
                param_norm = param.data.float().norm()
                update_ratio = param_diff.norm() / (param_norm + 1e-7)
                param_changes[name] = {
                    'update_norm': param_diff.norm().item(),
                    'param_norm': param_norm.item(),
                    'update_ratio': update_ratio.item()
                }
        
        # 计算整体更新统计
        total_update_norm = sum(change['update_norm'] for change in param_changes.values())
        total_param_norm = sum(change['param_norm'] for change in param_changes.values())
        avg_update_ratio = total_update_norm / (total_param_norm + 1e-7)
        
        swanlab.log({"text": swanlab.Text(f'Step stats - Grad norm: {grad_norm.item():.6f}, '
                  f'Avg update ratio: {avg_update_ratio:.6f}, '
                  f'Total update norm: {total_update_norm:.6f}')})
    lr_scheduler.step()
    


    return total_loss, grad_norm.item(), model_pred_norm,loss_no_weight,loss_no_logvar


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize global_step as 1
    global_step = 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    if args.seed is not None:
        set_seed(args.seed + rank)
    noise_random_generator = None

    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")

    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    teacher_transformer = deepcopy(transformer)


    if args.use_lora:
        assert args.model_type == "mochi", "LoRA is only supported for Mochi model."
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        transformer.add_adapter(transformer_lora_config)

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M")
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    print(fsdp_kwargs)

    if args.use_lora:
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        transformer._no_split_modules = no_split_modules
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)

    # transformer = FSDP(
    #     transformer,
    #     **fsdp_kwargs,
    # )
    
    # teacher_transformer = FSDP(
    #     teacher_transformer,
    #     **fsdp_kwargs,
    # )
    
    text_encoder = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device=torch.device('cuda'),
                checkpoint_path='/run/determined/workdir/data/H800/diffusion/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth',
                tokenizer_path='/run/determined/workdir/data/H800/diffusion/models/Wan2.1-T2V-1.3B/google/umt5-xxl',
                shard_fn=None)
    
    main_print("--> text-encoder loaded")

    
    transformer = transformer.cuda()
    teacher_transformer = teacher_transformer.cuda()

    main_print("--> model loaded")

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules, args.selective_checkpointing)
        apply_fsdp_checkpointing(teacher_transformer, no_split_modules, args.selective_checkpointing)


    transformer.train()
    teacher_transformer.requires_grad_(False)


    # Initialize SCM scheduler
    noise_scheduler = SCMScheduler()

    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(transformer, args.resume_from_lora_checkpoint,
                                                                   optimizer)
    main_print(f"optimizer: {optimizer}")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

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

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps * args.sp_size / args.train_sp_batch_size)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        swanlab.init(project='distill-scm', config=args , mode='cloud',logdir="/home/tempuser11/botehuang/swanlab")

    total_batch_size = (world_size * args.gradient_accumulation_steps / args.sp_size * args.train_sp_batch_size)
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        disable=rank > 0,
    )
    global_step = init_steps+1

    # loader = sp_parallel_dataloader_wrapper(train_dataloader, args.sp_size)
    loader = sp_parallel_dataloader_wrapper(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )
    
    writer = SummaryWriter(log_dir="./logs")


    for epoch in range(args.num_train_epochs):
        train_loss = 0.0
        for step in range(num_update_steps_per_epoch):
            # Skip steps for resuming
            if global_step < init_steps:
                progress_bar.update(1)
                global_step += 1
                continue

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, grad_norm, model_pred_norm,loss_no_weight,loss_no_logvar  = distill_one_step(
                    transformer=transformer,
                    model_type=args.model_type,
                    teacher_transformer=teacher_transformer,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    loader=loader,
                    noise_scheduler=noise_scheduler,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    sp_size=args.sp_size,
                    max_grad_norm=args.max_grad_norm,
                    uncond_prompt_embed=uncond_prompt_embed,
                    uncond_prompt_mask=uncond_prompt_mask,
                    not_apply_cfg_solver=args.not_apply_cfg_solver,
                    distill_cfg=args.distill_cfg,
                    hunyuan_teacher_disable_cfg=args.hunyuan_teacher_disable_cfg,
                    global_step=global_step,
                    text_encoder = text_encoder,
                    writer = writer,

                )
                # Increment global_step after each distill step
                global_step += 1

            train_loss += loss

            logs = {
                "loss": loss,
                "loss_no_weight":loss_no_weight,
                "loss_no_logvar":loss_no_logvar,
                "lr": lr_scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "step": global_step,
    
            }
            logs.update(model_pred_norm)
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            if rank <= 0:
                swanlab.log(logs)

            # if global_step % args.checkpointing_steps == 0:
            if global_step in [2,5,100,300,500,1000,2000,3000,4000,5000]:
                if rank <= 0:
                    if args.use_lora:
                        main_print(f"  save lora checkpoint in step：{global_step}")
                        save_lora_checkpoint(
                            transformer,
                            optimizer,
                            args.output_dir,
                            global_step,
                        )
                    else:
                        main_print(f"  save checkpoint in step：{global_step}")
                        save_checkpoint(
                            transformer,
                            rank,
                            args.output_dir,
                            step=global_step,
                            use_fsdp = False

                        )
                        

    # Save final checkpoint
    if rank <= 0:
        if args.use_lora:
            main_print(f"  save final lora checkpoint in step：{args.max_train_steps}")
            save_lora_checkpoint(
                transformer,
                optimizer,
                args.output_dir,
                args.max_train_steps,
            )
        else:
            main_print(f"  save final checkpoint in step：{args.max_train_steps}")
            save_checkpoint(
                transformer,
                rank,
                args.output_dir,
                args.max_train_steps,
                use_fsdp = False

            )

    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--cfg_embed', type=bool, default=True,
                    help='Whether to use configuration embedding')
    parser.add_argument('--cfg_embed_scale', type=float, default=0.1,
                    help='Scale factor for configuration embedding')
    parser.add_argument('--sample_neg_prompt', type=str, default='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
                   help='sample_neg_prompt')
    
    # sCM   training arguments
    parser.add_argument('--tangent_warmup_steps', type=int, default=1,
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
    parser.add_argument("--data_json_path", type=str , default= "/run/determined/workdir/data/H800/datasets/webvid-10k/Image-Vid-wan/videos2captionclip.json")
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
    parser.add_argument("--num_latent_t", type=int, default=2, help="Number of latent timesteps.")
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/run/determined/workdir/data/H800/datasets/webvid-10k/outputs/1e5-latent2-rejvp-cfg5-norm",
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
        default=5,
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
        default=3000,
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
        default=1e-5,
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
    parser.add_argument("--max_grad_norm", default=1, type=float, help="Max gradient norm.")
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
    parser.add_argument("--distill_cfg", type=float, default=5.0, help="Distillation coefficient.")
    # ["euler_linear_quadratic", "pcm", "pcm_linear_qudratic"]
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
        help="Weight type to use - fp32 or bf16.",
    )
    args = parser.parse_args()
    main(args)
