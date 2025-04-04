export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline

DATA_DIR=/storage/lintaoLab/lintao/botehuang/datasets/demotest
IP=[MASTER NODE IP]

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node 1\
    --rdzv-endpoint=localhost:29516\
    video_sana/distill_scm.py
    # --data_json_path "/storage/lintaoLab/lintao/botehuang/datasets/demotest/Image-Vid-Finetune-stepvideo/videos2caption.json" \
    # --seed 42\
    # --pretrained_model_name_or_path /storage/lintaoLab/lintao/botehuang/diffusion/models/stepvideo-t2v\
    # --dit_model_name_or_path /storage/lintaoLab/lintao/botehuang/diffusion/models/stepvideo-t2v/transformers/mp_rank_00_model_states.pt\
    # --model_type "stepvideo" \
    # --cache_dir "$DATA_DIR/.cache"\
    
    # --validation_prompt_dir "$DATA_DIR/Image-Vid-Finetune-stepvideo/validation"\
    # --gradient_checkpointing\
    # --train_batch_size=1\
    # --num_latent_t 32 \
    # --sp_size 2 \
    # --train_sp_batch_size 1\
    # --dataloader_num_workers 4\
    # --gradient_accumulation_steps=1\
    # --max_train_steps=320\
    # --learning_rate=1e-6\
    # --mixed_precision="bf16"\
    # --checkpointing_steps=64\
    # --validation_steps 64\
    # --validation_sampling_steps "2,4,8" \
    # --checkpoints_total_limit 3\
    # --allow_tf32\
    # --ema_start_step 0\
    # --cfg 0.0\
    # --log_validation\
    # --output_dir="$DATA_DIR/outputs/step_phase1_shift17_bs_16_HD"\
    # --tracker_project_name StepVideo_Distill \
    # --num_height 720 \
    # --num_width 1280 \
    # --num_frames  136 \
    # --shift 17 \
    # --validation_guidance_scale "1.0" \
    # --num_euler_timesteps 50 \
    # --multi_phased_distill_schedule "4000-1" \
    # --not_apply_cfg_solver 