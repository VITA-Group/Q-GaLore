# LLaMA-7B, QGaLore-Adam8bit, 8 A100, 8 Node
export NCCL_P2P_DISABLE=1
torchrun --standalone --nnodes 8 --nproc_per_node 8 run_pretrain.py \
    --model_config configs/llama_7b.json \
    --lr 0.004 \
    --galore_scale 0.25 \
    --rank 1024 \
    --update_proj_gap 500 \
    --batch_size 8 \
    --total_batch_size 512 \
    --num_training_steps 150000 \
    --warmup_steps 15000 \
    --weight_decay 0 \
    --grad_clipping 1.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer q_galore_adamw8bit \
    --project 'g-galore-c4' \
    --weight_quant \
    --stochastic_round \
    --proj_quant \
    --name Q-Galore-Adam8bit-LLaMA-7B