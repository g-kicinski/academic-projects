# Efficient Multi-GPU Transformer Pretraining
This repository contains the complete codebase, Slurm batch scripts, and experimental reports for the Transformer pretraining optimization assignment. The project transitions a basic single-GPU float32 C4 pretraining script into an efficient, distributed training pipeline featuring Fully Sharded Data Parallel (FSDP), Mixed Precision, Warmup-Stable-Decay (WSD) learning rate scheduling, SwiGLU activations, and a Chinchilla scaling laws grid search.

## Repository Structure

```
├── main.py                  # Core pretraining script with ArgumentParser, FSDP, WSD, and SwiGLU
├── requirements.txt         # Python dependencies
├── submit_begin.sub         # Basic single-GPU sbatch script (Step 1)
├── submit_torchrun.sub      # Distributed torchrun sbatch setup (Step 4.1)
├── submit_2gpu.sub          # Multi-GPU (2 GPUs, 1 node) FSDP training script (Step 4.3)
└── sbatch_grid_search.sub   # Slurm job array for Chinchilla grid search across nodes (Step 7)
```

## Implemented Features & Architecture


1.   CLI Arguments (`ArgumentParser`): Hyperparameters (`n_layers`, `dmodel`, `n_heads`, `batch_size`, `n_training_steps`, etc.) are configurable directly from the command line without modifying source files.
2.   Experiment Tracking: Integrated online monitoring (Weights & Biases / Neptune) logging training loss, learning rate schedules, and CLI configurations as run tags.
3. Distributed Training (FSDP): Configured using PyTorch FSDP with Mixed Precision (`bfloat16`/`float16` for `param_dtype` and `reduce_dtype`) and modified data loaders to ensure each device processes unique data batches across nodes.
4. WSD Learning Rate Scheduler: Implemented the Warmup-Stable-Decay schedule where warmup spans 1% and decay spans 10% of total training steps.
5. SwiGLU Activation: Replaced standard activation functions in the Feed-Forward Network layers with SwiGLU for enhanced representation capacity.
6. Chinchilla Scaling & Grid Search: Computed the compute-optimal token count ($D = 20N$) and executed a Slurm job array grid search over learning rates (`[1e-2, 1e-3, 1e-4]`) using multi-node setup (`--nodes 2 --gpus 2`).

## Running Jobs via Slurm
All job configurations are saved as automated .sub scripts.
```
sbatch submit_begin.sub        # Basic test run
sbatch submit_2gpu.sub         # 2-GPU FSDP training run
sbatch sbatch_grid_search.sub  # Chinchilla scaling grid search array
```