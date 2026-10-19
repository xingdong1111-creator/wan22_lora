"""
通用视频数据集准备工具: 把「视频目录 + 标注文件」组织成 DiffSynth-Studio 训练格式并质检。

支持标注格式: csv / json / jsonl, 字段自动识别:
  视频列: video / path / file / video_path / clip / video_name
  描述列: prompt / caption / text / description
  评分列(可选筛): aesthetic / motion / temporal / quality ...

用法:
  python prepare_dataset.py --src <视频所在目录> --anno <标注文件> --out <输出数据集目录> \
      [--score-key aesthetic --score-min 4.0] [--min-frames 16] [--max-items 2000] [--link] [--dry-run]

产出:
  out/metadata.csv   (video,prompt 两列, DiffSynth 格式, video 为相对 out 的路径)
  out/videos/        (合格视频, 复制或软链)
  终端打印质检报告(合格/淘汰及原因)
"""
import os, sys, json, csv, shutil, argparse

VIDEO_KEYS = ["video", "path", "file", "video_path", "clip", "video_name"]
PROMPT_KEYS = ["prompt", "caption", "text", "description", "cap"]
SCORE_KEYS = ["aesthetic", "aesthetics", "aesthetic_score", "motion", "motion_score",
              "temporal", "temporal_score", "quality", "quality_score"]


def pick_key(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return k
    return None


def load_anno(path):
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("annotations", list(data.values())[0]))
        return data
    if path.endswith(".jsonl"):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    import pandas as pd
    df = pd.read_csv(path)
    return [df.iloc[i].to_dict() for i in range(len(df))]


def probe_video(path, min_frames, min_h, min_w):
    """质检单个视频: 可读/帧数够/分辨率够。返回 (ok, reason, n, h, w)。"""
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio
    try:
        reader = imageio.get_reader(path, "ffmpeg")
        n, h, w = 0, 0, 0
        for f in reader:
            if n == 0:
                h, w = f.shape[0], f.shape[1]
            n += 1
            if n >= min_frames:
                break
        reader.close()
    except Exception as e:
        return False, f"读失败:{e}", 0, 0, 0
    if n < min_frames:
        return False, f"帧数{n}<{min_frames}", n, h, w
    if h < min_h or w < min_w:
        return False, f"分辨率{w}x{h}太小", n, h, w
    return True, "ok", n, h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="视频所在目录")
    ap.add_argument("--anno", required=True, help="标注文件 csv/json/jsonl")
    ap.add_argument("--out", required=True, help="输出数据集目录")
    ap.add_argument("--score-key", default=None, help="按该评分列筛选(如 aesthetic)")
    ap.add_argument("--score-min", type=float, default=None, help="评分下限")
    ap.add_argument("--min-frames", type=int, default=16)
    ap.add_argument("--min-height", type=int, default=240)
    ap.add_argument("--min-width", type=int, default=240)
    ap.add_argument("--max-items", type=int, default=None, help="最多保留条数")
    ap.add_argument("--link", action="store_true", help="软链而非复制视频(省空间)")
    ap.add_argument("--dry-run", action="store_true", help="只质检不写文件")
    args = ap.parse_args()

    rows = load_anno(args.anno)
    print(f"标注共 {len(rows)} 条")
    if not rows:
        print("❌ 标注为空")
        sys.exit(2)
    vk = pick_key(rows[0], VIDEO_KEYS)
    pk = pick_key(rows[0], PROMPT_KEYS)
    if not vk or not pk:
        print(f"❌ 识别不到视频/描述列, 现有列: {list(rows[0].keys())}")
        sys.exit(2)
    sk = args.score_key or pick_key(rows[0], SCORE_KEYS)
    print(f"视频列={vk} 描述列={pk} 评分列={sk}")

    vdir = os.path.join(args.out, "videos")
    if not args.dry_run:
        os.makedirs(vdir, exist_ok=True)

    kept, dropped = [], []
    for r in rows:
        name = os.path.basename(str(r[vk]))
        vpath = os.path.join(args.src, str(r[vk]))
        if not os.path.exists(vpath):
            dropped.append((name, "文件不存在"))
            continue
        if sk and args.score_min is not None:
            try:
                if float(r[sk]) < args.score_min:
                    dropped.append((name, f"评分{r[sk]}<{args.score_min}"))
                    continue
            except Exception:
                pass
        ok, reason, n, h, w = probe_video(vpath, args.min_frames, args.min_height, args.min_width)
        if not ok:
            dropped.append((name, reason))
            continue
        kept.append((name, str(r[pk])))
        if args.max_items and len(kept) >= args.max_items:
            break

    print(f"\n合格 {len(kept)} 条, 淘汰 {len(dropped)} 条")
    for name, reason in dropped[:20]:
        print(f"  淘汰: {name} ({reason})")
    if len(dropped) > 20:
        print(f"  ... 其余 {len(dropped) - 20} 条略")

    if args.dry_run:
        print("\n(dry-run, 未写文件)")
        return
    if not kept:
        print("❌ 无合格数据")
        sys.exit(2)

    with open(os.path.join(args.out, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["video", "prompt"])
        for name, prompt in kept:
            dst = os.path.join(vdir, name)
            srcp = os.path.join(args.src, name)
            if not os.path.exists(dst):
                if args.link:
                    os.symlink(os.path.abspath(srcp), dst)
                else:
                    shutil.copy2(srcp, dst)
            wtr.writerow([f"videos/{name}", prompt])
    print(f"\n✅ 已生成 {os.path.join(args.out, 'metadata.csv')} ({len(kept)} 条) + {vdir}/")


if __name__ == "__main__":
    main()
