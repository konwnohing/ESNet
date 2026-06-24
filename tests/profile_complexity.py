import os
import sys
import time
import csv
import torch

try:
    from thop import profile
except ImportError:
    raise ImportError("请先安装 thop: pip install thop")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_stereonet import EnhancedStereoNet


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_output_disp(outputs):
    """
    只用于 warm-up / timing，确保模型完整前向跑完。
    """
    if isinstance(outputs, dict):
        if "disp_final" in outputs:
            return outputs["disp_final"]
        if "disp_init" in outputs:
            return outputs["disp_init"]
        for v in outputs.values():
            if torch.is_tensor(v):
                return v
            if isinstance(v, dict):
                for vv in v.values():
                    if torch.is_tensor(vv):
                        return vv
    elif isinstance(outputs, (list, tuple)):
        for x in reversed(outputs):
            if torch.is_tensor(x):
                return x
    elif torch.is_tensor(outputs):
        return outputs

    raise RuntimeError("No tensor output found.")


def profile_one_model(name, use_prn, use_gate, h, w, device, warmup=50, iters=100):
    print("\n" + "=" * 80)
    print(f"Profiling: {name}")
    print(f"use_prn={use_prn}, use_gate={use_gate}, input={h}x{w}")
    print("=" * 80)

    model_cpu = EnhancedStereoNet(
        backbone_type="resnet18",
        max_disp=192,
        use_prn=use_prn,
        use_gate=use_gate,
        debug=False
    ).cpu().eval()

    total_params, trainable_params = count_params(model_cpu)

    dummy_l_cpu = torch.randn(1, 3, h, w).cpu()
    dummy_r_cpu = torch.randn(1, 3, h, w).cpu()

    print("[1/2] Counting Params and MACs...")
    try:
        macs, params_thop = profile(
            model_cpu,
            inputs=(dummy_l_cpu, dummy_r_cpu),
            verbose=False
        )
        macs_g = macs / 1e9
    except Exception as e:
        print(f"[Warning] THOP failed for {name}: {e}")
        macs_g = float("nan")

    params_m = total_params / 1e6
    trainable_m = trainable_params / 1e6

    print(f"Params: {params_m:.3f} M")
    print(f"Trainable Params: {trainable_m:.3f} M")
    print(f"MACs: {macs_g:.3f} G")

    del model_cpu
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    latency_ms = None
    fps = None
    peak_mem_mb = None

    if device.type == "cuda":
        print("[2/2] Measuring PyTorch CUDA latency...")

        torch.backends.cudnn.benchmark = True

        model_gpu = EnhancedStereoNet(
            backbone_type="resnet18",
            max_disp=192,
            use_prn=use_prn,
            use_gate=use_gate,
            debug=False
        ).to(device).eval()

        dummy_l = torch.randn(1, 3, h, w).to(device)
        dummy_r = torch.randn(1, 3, h, w).to(device)

        torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for _ in range(warmup):
                    out = model_gpu(dummy_l, dummy_r)
                    _ = get_output_disp(out)

        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        times = []

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for _ in range(iters):
                    starter.record()
                    out = model_gpu(dummy_l, dummy_r)
                    _ = get_output_disp(out)
                    ender.record()
                    torch.cuda.synchronize()
                    times.append(starter.elapsed_time(ender))

        latency_ms = sum(times) / len(times)
        fps = 1000.0 / latency_ms
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        print(f"PyTorch CUDA latency: {latency_ms:.2f} ms")
        print(f"PyTorch CUDA FPS: {fps:.2f}")
        print(f"Peak CUDA memory: {peak_mem_mb:.2f} MB")

        del model_gpu
        torch.cuda.empty_cache()

    else:
        print("[2/2] CUDA not available. Skip GPU timing.")

    return {
        "model": name,
        "use_prn": use_prn,
        "use_gate": use_gate,
        "input_size": f"{h}x{w}",
        "params_m": params_m,
        "trainable_params_m": trainable_m,
        "macs_g": macs_g,
        "pytorch_cuda_latency_ms": latency_ms,
        "pytorch_cuda_fps": fps,
        "peak_cuda_memory_mb": peak_mem_mb,
    }


def main():
    # 这里必须和论文里的部署/复杂度口径统一
    # 如果你最终部署表用 320x640，就改成 H=320, W=640
    H, W = 384,1248

    save_path = os.path.join(PROJECT_ROOT, "results", "model_complexity_profile.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_settings = [
        {
            "name": "Fast / Baseline",
            "use_prn": False,
            "use_gate": False,
        },
        {
            "name": "Balanced / PRN",
            "use_prn": True,
            "use_gate": False,
        },
        {
            "name": "Accurate / Full ESNet",
            "use_prn": True,
            "use_gate": True,
        },
    ]

    rows = []

    for m in model_settings:
        row = profile_one_model(
            name=m["name"],
            use_prn=m["use_prn"],
            use_gate=m["use_gate"],
            h=H,
            w=W,
            device=device,
            warmup=50,
            iters=100,
        )
        rows.append(row)

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 80)
    print("Profile finished.")
    print(f"Saved to: {save_path}")
    print("=" * 80)

    for r in rows:
        print(
            f"{r['model']:<25} | "
            f"Params: {r['params_m']:.3f}M | "
            f"MACs: {r['macs_g']:.3f}G | "
            f"Latency: {r['pytorch_cuda_latency_ms']} ms | "
            f"FPS: {r['pytorch_cuda_fps']}"
        )


if __name__ == "__main__":
    main()