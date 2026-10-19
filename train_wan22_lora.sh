#!/bin/bash
#SBATCH --job-name=wan22_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/wan22_lora_%j.out
#SBATCH --error=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/wan22_lora_%j.err

# === 环境设置 ===
# 不用 conda activate: 该集群 conda 激活有 IndexError bug, 会导致作业卡死。
# 直接把 wan22 环境的 bin 放到 PATH 最前, 等效于激活该环境。
export PATH="/public/home/liuhuan/anaconda/anaconda3/envs/wan22/bin:$PATH"
export CONDA_NO_PLUGINS=true
export PYTHONUNBUFFERED=1   # 实时刷新输出, 方便 tail 看日志

# === 工作目录 ===
cd /public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio

# === 创建日志目录 ===
mkdir -p logs

# === 环境变量 ===
# 预训练底模目录(全局共享,只读): liuhuan/models
export DIFFSYNTH_MODEL_BASE_PATH="/public/home/liuhuan/models"
# 训练产出目录(集中存放): workspace_xd/ckpt/<项目名>
export OUTPUT_CKPT_DIR="/public/home/liuhuan/workspace_xd/ckpt/wan22_ti2v_lora_openvid"
mkdir -p "${OUTPUT_CKPT_DIR}"
# 底模已下载完毕, 跳过远程检查加速启动
export DIFFSYNTH_SKIP_DOWNLOAD=True

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

# === 训练命令 ===
# Wan2.2-TI2V-5B LoRA 训练 (V100-32GB 适配)
# 数据集: OpenVid-1M 高质量子集 (936 条真实视频, 每条带英文 caption), 位于 workspace_xd/data/openvid_subset。
#   由 build_openvid_subset.py 从 OpenVid_part84.zip 按 美学>=5.8 / 时序>=0.99 / 时长>=3s / 运动<=25 筛出,
#   格式与官方示例一致: metadata.csv 两列 video,prompt; video 为相对 base_path 的 videos/xxx.mp4。
#   batch 固定为 1 -> 886 步/epoch (936 条按 seed=42 切走 50 条 test 后); num_epochs=2 共 ~1772 步; V100 实测 ~13s/it, 全程约 7~9h。
#   train/test 划分: metadata.csv=train 886 条(本脚本读取); metadata_test.csv=test 50 条(与 train 零重叠, train 不见),
#   供训练后评估泛化; metadata_all936.csv=划分前 936 条备份。
#   save_steps=300: 每 ~300 步存一个 step-*.safetensors (与按 epoch 存互斥), 便于早测推理并抗作业中断。
# - 使用 float16 代替 bfloat16 (V100 不支持 bf16)
# - 自动 fallback 到 PyTorch SDPA (无需 flash_attn)
# - 开启 gradient checkpointing 省显存
# - 开启 model CPU offload: DiT+T5+VAE 全常驻单张 V100-32GB 会在首个 step OOM
#   (表现为 CUBLAS_STATUS_NOT_INITIALIZED), 权重留 CPU、逐层上卡可规避
# - 单卡训练
# - 精度方案 (V100 fp16 关键, 解决之前的权重全 NaN 假成功): 底模用 --torch_dtype float16
#   省显存; LoRA 可训练参数已在 training_module.py 改为 upcast 到 fp32 (仅当 torch_dtype=fp16)。
#   再配合下面 accelerate 的 --mixed_precision fp16 (启用 GradScaler), 构成标准 AMP:
#   fp16 前向省显存+走 tensor core, fp32 LoRA 主权重 + GradScaler 防反向梯度下溢, Adam 更新稳定不 NaN。
#   注意: 两处改动必须成对! 若 LoRA 参数仍是 fp16, 会重现两个老 bug ——
#   不加 mixed_precision 则纯 fp16 梯度下溢致权重全 NaN(假成功); 加了则 optimizer.step 报
#   "ValueError: Attempting to unscale FP16 gradients"。upcast 到 fp32 后二者同时解决。

accelerate launch --num_processes 1 --mixed_precision fp16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path /public/home/liuhuan/workspace_xd/data/openvid_subset \
  --dataset_metadata_path /public/home/liuhuan/workspace_xd/data/openvid_subset/metadata.csv \
  --height 480 --width 832 --num_frames 49 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --save_steps 300 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_CKPT_DIR}" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing \
  --enable_model_cpu_offload \
  --torch_dtype float16
TRAIN_EXIT=$?

echo "=========================================="
echo "End: $(date)  (accelerate exit code: ${TRAIN_EXIT})"
echo "=========================================="
# 传播 accelerate 退出码: 训练崩溃时让 SLURM 记为 FAILED, 避免"假 COMPLETED"
exit ${TRAIN_EXIT}
