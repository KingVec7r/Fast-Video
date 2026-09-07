QWEN_3="Qwen/Qwen3-0.6B"
FAST_VIT_HD="kevin510/fast-vit-hd"

DATA_PATH="/path/to/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
VERSION="plain"
MM_VERSION="compressor"
CHECK_POINT="checkpoints/checkpoints_pretrain/checkpoint-1090"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

deepspeed --include localhost:0,1 --master_port 13354 fast_onevision/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${QWEN_3} \
    --mm_version ${MM_VERSION} \
    --mm_vision_tower ${FAST_VIT_HD} \
    --data_path ${DATA_PATH} \
    --conversation_format ${VERSION} \
    --adapter_checkpoint ${CHECK_POINT} \
    --data_path ${DATA_PATH} \
    --fixed_image_token_num 16 \
    --mm_max_compress_loop 1024 \
    --compress_intent_modeler_layer_num 2 \
    --compress_intent_token_num 4 \
    --compress_guidance_gen_layer_num 2 \
    --compress_layer_num 2 \
    --group_strategy length \
    --bf16 True \
    --output_dir ./checkpoints/checkpoints_pretrain_compressor \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --lr_query 1e-3 \
    --max_grad_norm 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 64 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --report_to none \