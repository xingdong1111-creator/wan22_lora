"""
Wan2.2-TI2V-5B  训练后LoRA  vs  训练前底模  对比推理 (test集, V100-32GB适配)

用途: 在隔离的 test 集 (训练时模型没见过的样本) 上, 用相同的首帧/prompt/seed,
      分别用 [原始底模] 和 [训练后的 LoRA step-1772] 各生成一组视频, 直接对比 LoRA 学到了什么。
      底模只加载一次: 先跑 base 组, 再 load_lora 跑 lora 组, 两组严格可比。

V100 关键适配 (与 infer_wan22_base.py 一致):
  1. torch_dtype=float16 (V100 不支持 bf16)
  2. redirect_common_files=False (离线: 否则 T5/VAE 被重定向到不存在的转换版)
  3. 低显存 vram 管理: 权重 offload 到 CPU 内存, 逐层上卡
  4. tiled=True: VAE 分块解码省显存
  5. device="cuda:0" 带索引 (torch 2.4.0 mem_get_info 不接受无索引的 "cuda")
"""
import os
import csv
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# === 路径配置 ===
LORA_PATH = "/public/home/liuhuan/workspace_xd/ckpt/wan22_ti2v_lora_openvid/step-1772.safetensors"
DATA_ROOT = "/public/home/liuhuan/workspace_xd/data/openvid_subset"
TEST_CSV = os.path.join(DATA_ROOT, "metadata_test.csv")
OUT_DIR = "/public/home/liuhuan/workspace_xd/ckpt/wan22_ti2v_lora_openvid/inference_compare"
os.makedirs(OUT_DIR, exist_ok=True)

# === 对比样本数 (test 集前 N 条) + 生成参数 (对齐训练: 480x832, 49 帧) ===
NUM_SAMPLES = 3
HEIGHT, WIDTH, NUM_FRAMES, SEED = 480, 832, 49, 0
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# === 低显存 vram 管理配置: 全部 float16, 空闲权重 offload 到 CPU 内存 ===
vram_config = {
    "offload_dtype": torch.float16,
    "offload_device": "cpu",
    "onload_dtype": torch.float16,
    "onload_device": "cpu",
    "preparing_dtype": torch.float16,
    "preparing_device": "cuda:0",
    "computation_dtype": torch.float16,
    "computation_device": "cuda:0",
}

# === 读 test 样本 (video 为相对 DATA_ROOT 的路径, prompt 为英文 caption) ===
with open(TEST_CSV, newline="", encoding="utf-8") as f:
    samples = list(csv.DictReader(f))[:NUM_SAMPLES]
print(f">>> 选用 {len(samples)} 条 test 样本做对比", flush=True)

# === 加载底模 (整个作业只加载一次) ===
print(">>> 加载底模 Wan2.2-TI2V-5B (float16, 离线, redirect_common_files=False) ...", flush=True)
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.float16,
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    redirect_common_files=False,
    vram_limit=torch.cuda.mem_get_info(0)[1] / (1024 ** 3) - 2,
)


def extract_first_frame(video_path):
    """用 imageio 抽视频首帧作为 I2V 的输入图; 失败返回 None。"""
    try:
        import imageio
        reader = imageio.get_reader(video_path, "ffmpeg")
        frame = reader.get_data(0)
        reader.close()
        return Image.fromarray(frame)
    except Exception as e:
        print(f"!!! imageio(ffmpeg) 抽帧失败: {e}", flush=True)
    try:
        import imageio
        frames = imageio.mimread(video_path, memtest=False)
        return Image.fromarray(frames[0])
    except Exception as e:
        print(f"!!! imageio.mimread 抽帧失败: {e}", flush=True)
    return None


def run_group(tag):
    """对每条 test 样本: 抽首帧 + 用其 caption 做 I2V, 保存为 {tag}_{i}.mp4"""
    for i, s in enumerate(samples):
        vp = os.path.join(DATA_ROOT, s["video"])
        img = extract_first_frame(vp)
        if img is None:
            print(f"!!! 样本{i} 抽帧失败, 跳过: {vp}", flush=True)
            continue
        img = img.resize((WIDTH, HEIGHT))
        img.save(os.path.join(OUT_DIR, f"input_{i}.jpg"))
        print(f">>> [{tag}] 样本{i} 生成中: {s['video']}", flush=True)
        video = pipe(
            prompt=s["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            input_image=img,
            seed=SEED,
            tiled=True,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
        )
        out = os.path.join(OUT_DIR, f"{tag}_{i}.mp4")
        save_video(video, out, fps=15, quality=5)
        print(f">>> [{tag}] 样本{i} 已保存: {out}", flush=True)


# === 组1: 训练前底模 (无 LoRA) ===
print(">>> ===== 组1: BASE 原始底模 (无 LoRA) =====", flush=True)
run_group("base")

# === 组2: 训练后 LoRA (step-1772) ===
print(f">>> ===== 组2: LORA 训练后 ({os.path.basename(LORA_PATH)}) =====", flush=True)
pipe.load_lora(pipe.dit, LORA_PATH, alpha=1.0)
run_group("lora")

print(">>> 对比推理完成: base_* 与 lora_* 已就绪", flush=True)
