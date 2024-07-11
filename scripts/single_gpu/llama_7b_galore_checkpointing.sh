# LLaMA-7B, 8-bit GaLore-Adam, single GPU
# Memory cost: 22.02G, BSZ=16
torchrun --standalone --nproc_per_node 1 run_pretrain.py \
    --model_config configs/llama_7b.json \
    --lr 0.004 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 16 \
    --total_batch_size 512 \
    --activation_checkpointing \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --single_gpu \
    --optimizer galore_adamw8bit_per_layer
