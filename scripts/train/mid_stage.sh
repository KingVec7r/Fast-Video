QWEN_3="Qwen/Qwen3-0.6B"
FAST_VIT_HD="kevin510/fast-vit-hd"

DATA_PATH="/path/to/mid_stage.yaml"
VERSION="qwen3"
MM_VERSION="vlm"
CHECK_POINT="checkpoints/checkpoints_pretrain_compressor/checkpoint-1090"

RESUME_CHECKPOINT=""

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

deepspeed --include localhost:0,1 --master_port 13353 fast_onevision/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${QWEN_3} \
    --mm_version ${MM_VERSION} \
    --mm_vision_tower ${FAST_VIT_HD} \
    --data_path ${DATA_PATH} \
    --conversation_format ${VERSION} \
    --adapter_checkpoint ${CHECK_POINT} \
    --mm_max_compress_loop 1024 \
    --compress_intent_modeler_layer_num 2 \
    --compress_intent_token_num 4 \
    --compress_guidance_gen_layer_num 2 \
    --compress_layer_num 2 \
    --group_strategy image_token_num \
    --bf16 True \
    --output_dir ./checkpoints/checkpoints_mid \
    ${RESUME_CHECKPOINT:+--resume_from_checkpoint $RESUME_CHECKPOINT} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
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
    --model_max_length 1500 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --dataloader_prefetch_factor 2 \
    --seed 42 \
    --report_to none