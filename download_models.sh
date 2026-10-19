#!/bin/bash
# 预先下载 Wan2.2-TI2V-5B 训练所需模型权重
# 下载路径与 DiffSynth 训练时查找路径一致:
#   DIFFSYNTH_MODEL_BASE_PATH(./models) / <model_id> / <files>
# 训练时 require_downloading() 会忽略已存在文件, 因此不会重复下载

set -x

PYBIN="/public/home/liuhuan/anaconda/anaconda3/envs/wan22/bin"
MODELSCOPE="${PYBIN}/modelscope"
BASE="/public/home/liuhuan/models"

mkdir -p "${BASE}/Wan-AI/Wan2.2-TI2V-5B"
mkdir -p "${BASE}/Wan-AI/Wan2.1-T2V-1.3B"

# === [1/2] 主模型 Wan2.2-TI2V-5B ===
# 与 train_wan22_lora.sh 的 --model_id_with_origin_paths 完全一致:
#   diffusion_pytorch_model*.safetensors (DiT, 分片)
#   models_t5_umt5-xxl-enc-bf16.pth       (T5 文本编码器)
#   Wan2.2_VAE.pth                        (VAE)
echo "===== [1/2] downloading Wan-AI/Wan2.2-TI2V-5B ====="
"${MODELSCOPE}" download \
  --model Wan-AI/Wan2.2-TI2V-5B \
  --include "diffusion_pytorch_model*.safetensors" "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.2_VAE.pth" \
  --local_dir "${BASE}/Wan-AI/Wan2.2-TI2V-5B"

# === [2/2] tokenizer (train.py 第38行硬编码从 Wan2.1-T2V-1.3B 取) ===
echo "===== [2/2] downloading tokenizer google/umt5-xxl from Wan-AI/Wan2.1-T2V-1.3B ====="
"${MODELSCOPE}" download \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --include "google/umt5-xxl/*" \
  --local_dir "${BASE}/Wan-AI/Wan2.1-T2V-1.3B"

echo "===== DOWNLOAD FINISHED ====="
echo "--- file list ---"
find "${BASE}" -type f -exec ls -lh {} \;
echo "--- total size ---"
du -sh "${BASE}"
