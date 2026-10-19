#!/bin/bash
#SBATCH --job-name=wan22_infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/wan22_infer_%j.out
#SBATCH --error=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/wan22_infer_%j.err

# === 环境设置 (与 train_wan22_lora.sh 完全一致) ===
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
# 底模已下载完毕, 跳过远程检查(离线)
export DIFFSYNTH_SKIP_DOWNLOAD=True

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "=========================================="

# === 推理命令 ===
# 加载官方底模 Wan2.2-TI2V-5B + 训练产出的 LoRA(epoch-4), 生成一段视频做端到端验证。
# 单卡即可; 推理无梯度/优化器, 显存需求远低于训练, 但仍开低显存 offload 保险。
python infer_wan22_lora.py
INFER_EXIT=$?

echo "=========================================="
echo "End: $(date)  (python exit code: ${INFER_EXIT})"
echo "=========================================="
# 传播退出码: 推理崩溃时让 SLURM 记为 FAILED, 避免"假 COMPLETED"
exit ${INFER_EXIT}
