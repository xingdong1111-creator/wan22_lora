#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从下载好的 OpenVid-1M 分块中筛选高质量子集, 组织成 DiffSynth 训练格式。

背景:
  - OpenVid-1M.csv 表头: video,caption,aesthetic score,motion score,
    temporal consistency score,camera motion,frame,fps,seconds  (共 ~102 万行)
  - 视频被打包在 OpenVid_partN.zip 里, CSV 不含"属于哪个 zip"的列,
    所以流程是: 解压 zip -> 用视频文件名回 CSV 查 caption+评分 -> 按评分筛 -> 组织。

产出 (out 目录):
  out/metadata.csv      两列 video,prompt  (video="videos/<name>", prompt=caption)
  out/videos/<name>     被选中的视频文件 (物理拷贝, 因为训练在计算节点跑, 看不到 /tmp)

用法示例:
  python build_openvid_subset.py \
    --csv /tmp/openvid_dl/OpenVid-1M.csv \
    --src /tmp/openvid_dl/extracted \
    --out /public/home/liuhuan/workspace_xd/data/openvid_subset \
    --min-aesthetic 5.5 --min-temporal 0.95 --min-seconds 3.0 --top-n 2000
"""
import argparse
import csv
import os
import shutil
import sys

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# OpenVid-1M.csv 的确切列名 (注意 aesthetic/motion/temporal 中间是空格)
COL_VIDEO = "video"
COL_CAPTION = "caption"
COL_AES = "aesthetic score"
COL_MOTION = "motion score"
COL_TEMPORAL = "temporal consistency score"
COL_SECONDS = "seconds"


def to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_csv_index(csv_path):
    """读 OpenVid-1M.csv, 建 basename -> row 的索引。"""
    idx = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 校验表头是否含预期列
        fields = reader.fieldnames or []
        missing = [c for c in (COL_VIDEO, COL_CAPTION, COL_AES) if c not in fields]
        if missing:
            print("[警告] CSV 缺少预期列: %s ; 实际表头: %s" % (missing, fields))
        n = 0
        for row in reader:
            v = (row.get(COL_VIDEO) or "").strip()
            if not v:
                continue
            idx[os.path.basename(v)] = row
            n += 1
    print("[索引] 从 CSV 载入 %d 条标注" % n)
    return idx


def scan_videos(src):
    """递归扫描解压目录里的视频文件, 返回绝对路径列表。"""
    vids = []
    for root, _dirs, files in os.walk(src):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXT):
                vids.append(os.path.join(root, fn))
    vids.sort()
    return vids


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def report_scores(name, vals):
    if not vals:
        print("  %s: (无数据)" % name)
        return
    s = sorted(vals)
    print("  %-12s n=%d min=%.3f p25=%.3f 中位=%.3f p75=%.3f p90=%.3f max=%.3f"
          % (name, len(s), s[0], percentile(s, 0.25), percentile(s, 0.5),
             percentile(s, 0.75), percentile(s, 0.90), s[-1]))


def main():
    ap = argparse.ArgumentParser(description="OpenVid-1M 高质量子集筛选+组织工具")
    ap.add_argument("--csv", required=True, help="OpenVid-1M.csv 路径")
    ap.add_argument("--src", required=True, help="解压出的视频目录")
    ap.add_argument("--out", required=True, help="输出数据集目录(将建 metadata.csv + videos/)")
    ap.add_argument("--min-aesthetic", type=float, default=5.0, help="美学分下限")
    ap.add_argument("--min-temporal", type=float, default=0.90, help="时序一致性下限")
    ap.add_argument("--min-seconds", type=float, default=3.0, help="时长下限(秒)")
    ap.add_argument("--min-motion", type=float, default=0.0, help="运动分下限(过滤静止片段)")
    ap.add_argument("--max-motion", type=float, default=1e9, help="运动分上限(过滤过乱片段)")
    ap.add_argument("--top-n", type=int, default=2000, help="最多选多少条(按美学分降序)")
    ap.add_argument("--min-caption-len", type=int, default=20, help="caption 最短字符数")
    ap.add_argument("--dry-run", action="store_true", help="只统计+筛选, 不拷贝/不写文件")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print("[错误] CSV 不存在: %s" % args.csv)
        sys.exit(1)
    if not os.path.isdir(args.src):
        print("[错误] 视频目录不存在: %s" % args.src)
        sys.exit(1)

    idx = load_csv_index(args.csv)
    vids = scan_videos(args.src)
    print("[扫描] 解压目录中找到 %d 个视频文件" % len(vids))

    matched, unmatched = [], 0
    aes_all, mot_all, tmp_all = [], [], []
    for vp in vids:
        base = os.path.basename(vp)
        row = idx.get(base)
        if row is None:
            unmatched += 1
            continue
        aes = to_float(row.get(COL_AES))
        mot = to_float(row.get(COL_MOTION))
        tmp = to_float(row.get(COL_TEMPORAL))
        sec = to_float(row.get(COL_SECONDS))
        cap = (row.get(COL_CAPTION) or "").strip()
        if aes is not None:
            aes_all.append(aes)
        if mot is not None:
            mot_all.append(mot)
        if tmp is not None:
            tmp_all.append(tmp)
        matched.append({
            "path": vp, "name": base, "caption": cap,
            "aes": aes, "mot": mot, "tmp": tmp, "sec": sec,
        })

    print("[匹配] 命中 CSV %d 条, 未命中 %d 条" % (len(matched), unmatched))
    print("[分布] 命中视频的评分分布:")
    report_scores("aesthetic", aes_all)
    report_scores("motion", mot_all)
    report_scores("temporal", tmp_all)

    # 过滤
    passed = []
    for m in matched:
        if m["aes"] is None or m["aes"] < args.min_aesthetic:
            continue
        if args.min_temporal > 0 and (m["tmp"] is None or m["tmp"] < args.min_temporal):
            continue
        if m["sec"] is not None and m["sec"] < args.min_seconds:
            continue
        if m["mot"] is not None and not (args.min_motion <= m["mot"] <= args.max_motion):
            continue
        if len(m["caption"]) < args.min_caption_len:
            continue
        passed.append(m)

    # 按美学分降序, 取 top-n
    passed.sort(key=lambda x: (x["aes"] if x["aes"] is not None else -1), reverse=True)
    selected = passed[:args.top_n] if args.top_n and args.top_n > 0 else passed

    print("[筛选] 阈值(美学>=%.2f 时序>=%.2f 时长>=%.1fs 运动[%.2f,%.2f] caption>=%d字) -> 通过 %d 条, 选取 %d 条"
          % (args.min_aesthetic, args.min_temporal, args.min_seconds,
             args.min_motion, args.max_motion, args.min_caption_len,
             len(passed), len(selected)))
    if not selected:
        print("[错误] 没有视频通过筛选, 请放宽阈值。")
        sys.exit(2)
    if selected:
        print("[选取] 美学分范围: %.3f ~ %.3f" % (selected[-1]["aes"], selected[0]["aes"]))

    if args.dry_run:
        print("[dry-run] 不写文件。示例前 3 条:")
        for m in selected[:3]:
            print("  - %s | aes=%.3f | %s" % (m["name"], m["aes"], m["caption"][:80]))
        return

    # 拷贝到 out/videos + 写 metadata.csv
    out_videos = os.path.join(args.out, "videos")
    os.makedirs(out_videos, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")

    total_bytes = 0
    rows = []
    for i, m in enumerate(selected, 1):
        dst = os.path.join(out_videos, m["name"])
        if not os.path.exists(dst):
            shutil.copy2(m["path"], dst)
        total_bytes += os.path.getsize(dst)
        rows.append({"video": "videos/" + m["name"], "prompt": m["caption"]})
        if i % 200 == 0:
            print("  已拷贝 %d/%d ..." % (i, len(selected)))

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "prompt"])
        w.writeheader()
        w.writerows(rows)

    print("[完成] 写出 %d 条 -> %s" % (len(rows), meta_path))
    print("[完成] 视频总大小 %.2f GB -> %s" % (total_bytes / (1024**3), out_videos))
    print("[下一步] 训练脚本改: --dataset_base_path %s  --dataset_metadata_path %s"
          % (args.out, meta_path))


if __name__ == "__main__":
    main()
