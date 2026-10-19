#!/bin/bash
#SBATCH --job-name=wan22_base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/wan22_base_%j.out
#SBATCH --error=/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/logs/wan22_base_%j.err

# === 环境设置 (与 infer_wan22_lora.sh 完全一致) ===
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
# 纯底模 baseline: 只加载官方 Wan2.2-TI2V-5B, 不加任何 LoRA, 生成 base_i2v.mp4。
# 输入与 infer_wan22_lora.py 完全一致(prompt/首帧/seed/分辨率/帧数), 便于与 lora_i2v.mp4 直接对比。
python infer_wan22_base.py
INFER_EXIT=$?

echo "=========================================="
echo "End: $(date)  (python exit code: ${INFER_EXIT})"
echo "=========================================="
# 传播退出码: 推理崩溃时让 SLURM 记为 FAILED, 避免"假 COMPLETED"
exit ${INFER_EXIT}
