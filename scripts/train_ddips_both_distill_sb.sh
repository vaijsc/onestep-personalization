accelerate launch src/train_ddips_both_distill_sb.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --pretrained_sb_generator "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/all_backup_SP/ckpt/sbv2_sd1.5/0.7"\
    --pretrained_ip_adapter "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/all_backup_SP/ckpt/ip-adapter_sd15.bin" \
    --image_encoder_path "h94/IP-Adapter" \
    --num_ip_tokens 4 \
    --resume_from_checkpoint "latest" \
    --use_ema \
    --num_val_samples 5 \
    --resolution 512 \
    --validation_steps 5000 \
    --train_batch_size 6 \
    --gradient_accumulation_steps 1 --gradient_checkpointing \
	--set_grads_to_none \
    --guidance_scale 4.5 \
    --learning_rate 1.e-05 \
    --learning_rate_lora 1.e-03 \
    --lr_scheduler "cosine" \
    --lora_rank 64 --lora_alpha 108 \
	--adam_weight_decay 1.e-04 \
	--lr_warmup_steps 0 \
    --num_train_epochs 200 \
    --checkpointing_steps 20000 \
	--output_dir "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/tmp_results/train_ddips_both_sb_ffhq_cap" \
    --task "train_ffhq_cap"

# --pretrained_ip_adapter "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/ckpt/ip-adapter_sd15.bin" \