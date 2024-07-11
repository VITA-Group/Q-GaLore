# LLaMA-60M, QGaLore-Adam8bit, 1 A100, 1 Node
export NCCL_P2P_DISABLE=1
torchrun --standalone --nproc_per_node 1 run_pretrain.py \
    --model_config configs/llama_60m.json \
    --lr 0.015 \
    --galore_scale 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer q_galore_adamw8bit \
    --project 'g-galore-c4' \
    --weight_quant \
    --stochastic_round \
    --proj_quant \
    --name Q-Galore-Adam8bit-LLaMA-60M