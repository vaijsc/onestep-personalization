accelerate launch src/train_ip_adapter_sb2.py \
    --pretrained_model_name_or_path "stabilityai/sd-turbo" \
    --pretrained_sb_generator "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/sb_v2_ckpt/0.5"\
    --image_encoder_path "h94/IP-Adapter" \
    --resume_from_checkpoint "latest" \
    --use_ema \
    --resolution 512 \
    --validation_steps 20 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 1 --gradient_checkpointing \
	--set_grads_to_none \
    --guidance_scale 4.5 \
    --learning_rate 1.e-05 \
    --lr_scheduler cosine \
	--adam_weight_decay 1.e-04 \
	--lr_warmup_steps 0 \
    --num_train_epochs 200\
    --checkpointing_steps 20 \
	--output_dir "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/all_ckpt/train_ip_sb2_coca5k" \
    --task "coca5k"