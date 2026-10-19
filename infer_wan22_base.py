"""
Wan2.2-TI2V-5B 原始底模 baseline 推理脚本 (V100-32GB 适配)

用途: 只加载官方底模 Wan2.2-TI2V-5B, **不加载任何 LoRA**, 用与 infer_wan22_lora.py
      完全相同的输入 (prompt / 首帧图 / seed / 分辨率 / 帧数) 生成一段视频。
      作为"训练前"的基线, 与同目录下的 lora_i2v.mp4 直接对比, 一眼看出训练 LoRA 改变了什么;
      也用来确认原始 wan 底模本身能否正常出视频。

与 infer_wan22_lora.py 的唯一实质差别: 去掉 pipe.load_lora(...) 那一行 (保持原始底模权重)。

V100 关键适配 (与 infer_wan22_lora.py 一致):
  1. torch_dtype=float16 (V100 不支持 bfloat16, 官方示例用的是 bf16, 这里必须换)
  2. redirect_common_files=False (离线: 否则 T5/VAE 会被重定向到不存在的转换版 safetensors)
  3. 低显存 vram 管理: 权重 offload 到 CPU 内存, 逐层上卡 (对齐 model_inference_low_vram)
  4. tiled=True: VAE 分块解码省显存
"""
import os
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

# === 路径配置 (纯底模 baseline: 无 LORA_PATH) ===
TRAIN_VIDEO = "/public/home/liuhuan/workspace_xd/wan2_2_pro/DiffSynth-Studio/data/diffsynth_example_dataset/wanvideo/Wan2.2-TI2V-5B/video1.mp4"
# 输出到与 LoRA 推理相同的目录, 便于 base_*.mp4 与 lora_*.mp4 并排对比
OUT_DIR = "/public/home/liuhuan/workspace_xd/ckpt/wan22_ti2v_lora/inference"
os.makedirs(OUT_DIR, exist_ok=True)

# === 生成参数 (与 infer_wan22_lora.py 完全一致, 保证严格可比) ===
PROMPT = "from sunset to night, a small town, light, house, river"
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
HEIGHT, WIDTH, NUM_FRAMES, SEED = 480, 832, 49, 0

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

print(">>> 加载底模 Wan2.2-TI2V-5B (float16, 离线, redirect_common_files=False) ...", flush=True)
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.float16,
    # 注意: 必须用带索引的 "cuda:0", 库内 mem_get_info("cuda") 在 torch 2.4.0 会崩
    device="cuda:0",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    redirect_common_files=False,
    # 注意: torch 2.4.0 的 mem_get_info 不接受字符串 "cuda"(无索引), 必须传整数设备号 0
    vram_limit=torch.cuda.mem_get_info(0)[1] / (1024 ** 3) - 2,
)

# === 纯底模 baseline: 这里故意 **不** 调用 pipe.load_lora, 保持原始底模权重 ===
print(">>> [baseline] 不加载任何 LoRA, 使用原始底模权重直接生成", flush=True)


def extract_first_frame(video_path):
    """用 imageio 抽视频首帧作为 I2V 的输入图; 失败返回 None (回退到纯文生视频)。"""
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


input_image = extract_first_frame(TRAIN_VIDEO)
if input_image is not None:
    input_image = input_image.resize((WIDTH, HEIGHT))
    ref_path = os.path.join(OUT_DIR, "input_first_frame.jpg")
    input_image.save(ref_path)
    print(f">>> 已抽取训练视频首帧作为输入图: {ref_path} (I2V 模式)", flush=True)
    video = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        input_image=input_image,
        seed=SEED,
        tiled=True,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )
    out_path = os.path.join(OUT_DIR, "base_i2v.mp4")
else:
    print(">>> 无法抽帧, 回退到纯文生视频 (T2V 模式)", flush=True)
    video = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        seed=SEED,
        tiled=True,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )
    out_path = os.path.join(OUT_DIR, "base_t2v.mp4")

save_video(video, out_path, fps=15, quality=5)
print(f">>> 视频已保存: {out_path}", flush=True)
print(">>> baseline 推理完成 (纯底模, 未加 LoRA)", flush=True)
