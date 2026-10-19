#!/bin/bash
#SBATCH --job-name=smoke_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/smoke_lora_%j.out
#SBATCH --error=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/smoke_lora_%j.err

# ============================================================
# 冒烟测试 (非正式训练): 验证方案B修复了"权重全 NaN 假成功"
#   方案B = 底模 fp16 (--torch_dtype float16)
#         + LoRA 参数 upcast fp32 (training_module.py L341, 仅当 torch_dtype=fp16)
#         + accelerate --mixed_precision fp16 (GradScaler)
# 只跑 20 步 (dataset_repeat=20, num_epochs=1), 验证三件事:
#   1) optimizer.step() 不再报 "Attempting to unscale FP16 gradients"
#   2) loss.csv 里 loss 是有限值 (非 NaN)
#   3) epoch-0.safetensors 权重非 NaN
# 输出到独立临时目录, 不污染正式产出 wan22_ti2v_lora
# ============================================================

# === 环境设置 (同正式脚本: 不用 conda activate, 直接把 env/bin 放 PATH 最前) ===
export PATH="/public/home/liuhuan/anaconda/anaconda3/envs/wan22/bin:$PATH"
export CONDA_NO_PLUGINS=true
export PYTHONUNBUFFERED=1

cd /public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio
mkdir -p logs

# === 环境变量 ===
export DIFFSYNTH_MODEL_BASE_PATH="/public/home/liuhuan/models"
export OUTPUT_CKPT_DIR="/public/home/liuhuan/workspace_xd/ckpt/wan22_smoke_test"
mkdir -p "${OUTPUT_CKPT_DIR}"
export DIFFSYNTH_SKIP_DOWNLOAD=True

echo "=========================================="
echo "SMOKE TEST Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

# === 冒烟训练命令 (20 步) ===
accelerate launch --num_processes 1 --mixed_precision fp16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B \
  --dataset_metadata_path data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B/metadata.csv \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 20 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_CKPT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --enable_model_cpu_offload \
  --enable_csv_log \
  --torch_dtype float16
TRAIN_EXIT=$?

echo "=========================================="
echo "End: $(date)  (accelerate exit code: ${TRAIN_EXIT})"
echo "=== loss.csv (每步 loss, 应全为有限值) ==="
cat "${OUTPUT_CKPT_DIR}/loss.csv" 2>/dev/null || echo "(loss.csv 未生成)"
echo "=== 产出文件 ==="
ls -la "${OUTPUT_CKPT_DIR}" 2>/dev/null
echo "=========================================="
exit ${TRAIN_EXIT}
