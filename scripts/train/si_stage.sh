DATA_PATH="/path/to/si_stage.yaml"
VERSION="qwen3"
MM_VERSION="vlm"
CHECK_POINT="checkpoints/checkpoints_mid/checkpoint-7762"

RESUME_CHECKPOINT=""

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

deepspeed --include localhost:4,5,6,7 --master_port 13353 fast_onevision/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${CHECK_POINT} \
    --data_path ${DATA_PATH} \
    --conversation_format ${VERSION} \
    --mm_version ${MM_VERSION} \
    --mm_max_compress_loop 1024 \
    --group_strategy image_token_num \
    --bf16 True \
    --output_dir ./checkpoints/checkpoints_si \
    ${RESUME_CHECKPOINT:+--resume_from_checkpoint $RESUME_CHECKPOINT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --lr_vision_tower 2e-6 \
    --lr_projector 1e-5 \
    --lr_llm 1e-5 \
    --lr_query 1e-5 \
    --lr_compressor 1e-5 \
    --max_grad_norm 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --dataloader_prefetch_factor 2 \
    --seed 42 \
    --report_to none