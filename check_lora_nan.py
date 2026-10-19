"""检查 LoRA safetensors 权重是否含 NaN/Inf, 并报告 dtype 与数值范围。

用途: 训练产出的 LoRA 权重可能"假成功"(作业 COMPLETED 但数值全 NaN), 必须体检。
用法: python check_lora_nan.py <目录 或 .safetensors 文件> [更多路径...]
      目录会检查其下所有 *.safetensors。
判读: ✅OK = 无 NaN/Inf; ❌NaN! = 中招(训练数值发散); dtype 应为 ['torch.float32']。
"""
import sys, os, glob, torch
from safetensors.torch import load_file


def check(path):
    sd = load_file(path)
    vals = [v.flatten().float() for v in sd.values()]
    if not vals:
        print(f"[EMPTY] {path}")
        return False
    v = torch.cat(vals)
    nan = torch.isnan(v).sum().item()
    inf = torch.isinf(v).sum().item()
    dtypes = sorted(set(str(x.dtype) for x in sd.values()))
    ok = (nan == 0 and inf == 0)
    flag = "✅OK  " if ok else ("❌NaN!" if nan > 0 else "❌Inf!")
    print(
        f"{flag} {os.path.basename(path):28s} | 张量{len(sd):4d} 元素{v.numel():9d} "
        f"NaN={nan:9d} Inf={inf:6d} | min={v.min():+.5f} max={v.max():+.5f} "
        f"mean={v.mean():+.5f} std={v.std():.5f} | dtype={dtypes}"
    )
    return ok


def main():
    targets = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            targets += sorted(glob.glob(os.path.join(arg, "*.safetensors")))
        else:
            targets.append(arg)
    if not targets:
        print("无目标文件 (用法: python check_lora_nan.py <目录或文件>)")
        sys.exit(1)
    all_ok = True
    for t in targets:
        if os.path.exists(t):
            all_ok = check(t) and all_ok
        else:
            print(f"[缺失] {t}")
            all_ok = False
    print("\n总结:", "全部权重健康 ✅" if all_ok else "存在 NaN/Inf ❌ (训练数值发散)")
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
