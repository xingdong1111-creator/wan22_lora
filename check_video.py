"""
推理产出视频体检工具: 抽帧分析像素数值 + 帧间变化, 判定是否黑屏/静止。

用法:
    python check_video.py [视频路径]
默认检查 ckpt/wan22_ti2v_lora/inference/lora_i2v.mp4。

会把关键帧存到 workspace_xd/wan2_2_pro/frames/ 供直接查看画面。
"""
import sys, os
import numpy as np
try:
    import imageio.v2 as imageio
except Exception:
    import imageio

DEFAULT = "/public/home/liuhuan/workspace_xd/ckpt/wan22_ti2v_lora/inference/lora_i2v.mp4"
OUTDIR = "/public/home/liuhuan/workspace_xd/wan2_2_pro/frames"


def load_frames(path):
    """优先用 ffmpeg reader 逐帧读; 失败回退 mimread。返回 list[np.uint8 array]。"""
    try:
        reader = imageio.get_reader(path, "ffmpeg")
        meta = reader.get_meta_data()
        frames = [np.array(f) for f in reader]
        reader.close()
        return frames, meta
    except Exception as e:
        print(f"!!! ffmpeg reader 失败({e}), 回退 mimread", flush=True)
        return [np.array(f) for f in imageio.mimread(path, memtest=False)], {}


def check(path):
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return False
    frames, meta = load_frames(path)
    n = len(frames)
    h, w = frames[0].shape[0], frames[0].shape[1]
    print(f"视频: {os.path.basename(path)}")
    print(f"分辨率: {w}x{h}  总帧数: {n}  fps: {meta.get('fps', '?')}")

    os.makedirs(OUTDIR, exist_ok=True)
    idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    print("\n=== 关键帧像素数值 ===")
    for i in idxs:
        f = frames[i].astype(np.float32)
        print(f"帧{i:3d}: min={f.min():5.1f} max={f.max():5.1f} mean={f.mean():6.2f} std={f.std():6.2f}")
        imageio.imwrite(os.path.join(OUTDIR, f"frame_{i:03d}.jpg"), frames[i])
    print(f"(关键帧已存 {OUTDIR}/frame_*.jpg)")

    # 全片统计
    gmin = min(float(fr.min()) for fr in frames)
    gmax = max(float(fr.max()) for fr in frames)
    gmean = float(np.mean([fr.mean() for fr in frames]))
    gstd = float(np.mean([fr.std() for fr in frames]))
    # 帧间平均差异(检测静止)
    diffs = [float(np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32)).mean())
             for i in range(n - 1)]
    mdiff = float(np.mean(diffs)) if diffs else 0.0

    print(f"\n=== 全片({n}帧)统计 ===")
    print(f"全局 min={gmin:.1f} max={gmax:.1f} mean={gmean:.2f} std={gstd:.2f}")
    print(f"帧间平均差异={mdiff:.2f} (0=完全静止)")

    print("\n=== 判定 ===")
    ok = True
    if gmax < 30 and gmean < 5:
        print("❌ 黑屏: 整体亮度极低, 无画面内容")
        ok = False
    elif gmean < 5:
        print("⚠️ 极暗: 画面几乎全黑")
        ok = False
    else:
        print(f"✅ 非黑屏: mean={gmean:.1f} 有正常亮度; max={gmax:.0f} 有高光; std={gstd:.1f} 有细节层次")
        if mdiff < 0.5:
            print("⚠️ 但帧间几乎无变化(疑似静止/单帧重复)")
        else:
            print(f"✅ 帧间有变化(平均差异 {mdiff:.1f}), 是动态视频")
    return ok


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    good = check(p)
    sys.exit(0 if good else 2)
