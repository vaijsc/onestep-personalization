accelerate launch src/train_ddips_distill_ip_dmd2.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --pretrained_dmd_generator "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/checkpoints/dmdv2/sd15/pytorch_model.bin"\
    --pretrained_ip_adapter "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/ckpt/ip-adapter_sd15.bin" \
    --image_encoder_path "h94/IP-Adapter" \
    --num_ip_tokens 4 \
    --resume_from_checkpoint "latest" \
    --use_ema \
    --resolution 512 \
    --validation_steps 5000 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 --gradient_checkpointing \
	--set_grads_to_none \
    --guidance_scale 4.5 \
    --learning_rate 1.e-05 \
    --learning_rate_lora 1.e-03 \
    --lr_scheduler "constant" \
    --lora_rank 64 --lora_alpha 108 \
	--adam_weight_decay 1.e-04 \
	--lr_warmup_steps 0 \
    --num_train_epochs 200 \
    --checkpointing_steps 5000 \
	--output_dir "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/all_ckpt/train_ddips_distill_ip_dmd2_coca100k" \
    --task "coca100k"

# --pretrained_ip_adapter "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/ckpt/ip-adapter_sd15.bin" \