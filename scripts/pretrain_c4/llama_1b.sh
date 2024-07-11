# LLaMA-1B, QGaLore-Adam8bit, 8 A100, 1 Node
export NCCL_P2P_DISABLE=1
torchrun --standalone --nproc_per_node 8 run_pretrain.py \
    --model_config configs/llama_1b.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 200 \
    --batch_size 16 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer q_galore_adamw8bit \
    --project 'g-galore-c4' \
    --weight_quant \
    --stochastic_round \
    --proj_quant \
    --name Q-Galore-Adam8bit-LLaMA-1B