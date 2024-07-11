# LLaMA-350M, QGaLore-Adam8bit, 4 A100, 1 Node
# Simulation Mode of quantization
export NCCL_P2P_DISABLE=1
torchrun --standalone --nproc_per_node 4 run_pretrain.py \
    --model_config configs/llama_350m.json \
    --lr 0.01 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer q_galore_adamw8bit \
    --project 'g-galore-c4' \
    --weight_quant \
    --stochastic_round \
    --proj_quant \
    --simulation \
    --name Q-Galore-Adam8bit-LLaMA-350M