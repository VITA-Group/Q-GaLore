# Q-GaLore

This repo contains the pre-release version of Q-GaLore algorithm, proposed by [Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients](https://arxiv.org/abs/2407.08296).

Q-GaLore is a memory-efficient training methodology effective in both pre-training and fine-tuning scenarios. Q-GaLore incorporates two main components: (i) low precision training utilizing low-rank gradients, and (ii) lazy layer-wise subspace exploration. It enables full-parameter learning while requiring less memory, such as training a LLaMA-7B model from scratch on a single NVIDIA RTX 4060 Ti with only 16GB of memory.

<div align="center">
  <img src="imgs/q-galore.jpg" alt="Image 2" style="width: 550px; margin: 0 auto;">
</div>

Read this [blog](https://www.linkedin.com/pulse/introducing-galore-v2-q-galore-latest-milestone-low-rank-atlas-wang-lpijc/?trackingId=Yk7Uh3ptT0uoKE5TQwWPLA%3D%3D) for more details!

### Install Q-GaLore optimizer

##### Install via conda

```
conda env create -f environment.yml
```

##### or Install Q-GaLore optimizer and experiment dependencies

```bash
# install from pip
pip install q-galore-torch


# or install from source:
git clone https://github.com/VITA-Group/Q-GaLore.git
cd Q-GaLore
pip install -e .

pip install -r exp_requirements.txt
```

## Usage

##### Pretraining LLaMA model on C4 dataset

We provide the command in `scripts/pretrain_c4` for pretraining LLaMA model with sizes from 60M to 7B on C4 dataset. We also provide the simulation mode implementation of quantization with scripts in `scripts/pretrain_c4/simulation`.  For example, training a LLaMA-60M with Q-GaLore-Adam8bit with the following scripts.

```
torchrun --standalone --nproc_per_node 1 run_pretrain.py \
    --model_config configs/llama_130m.json \
    --lr 0.015 \
    --galore_scale 0.25 \
    --rank 256 \
    --update_proj_gap 200 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer q_galore_adamw8bit \
    --project 'g-galore-c4' \
    --weight_quant \
    --stochastic_round \
    --proj_quant \
    --name Q-Galore-Adam8bit-LLaMA-130M
```

##### Pretraining LLaMA-7B model within 16GB memory

The command of training LLaMA-7B model on single GPU as provided within `scripts/pretrain_c4/single_gpu`. With 16 batch size and activation checkpointing, the following scripts can pre-train a LLaMA-7B model with 15.26GB memory (tested on a single A6000 GPU)

```
# LLaMA-7B, 8-bit Q-GaLore-Adam, single GPU
# Memory cost: 15.26G, BSZ=16
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
    --proj_quant \
    --weight_quant \
    --stochastic_round \
    --optimizer q_galore_adamw8bit_per_layer

```

## Citation

```bibtex
@misc{zhang2024qgalore,
      title={Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients}, 
      author={Zhenyu Zhang and Ajay Jaiswal and Lu Yin and Shiwei Liu and Jiawei Zhao and Yuandong Tian and Zhangyang Wang},
      year={2024},
      eprint={2407.08296},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.08296}, 
}
```
