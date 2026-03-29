import os
import subprocess


def detect_gpu_layers(default_gpu_layers=-1):
    override = os.environ.get("GLM_GPU_LAYERS")
    if override is not None:
        try:
            return int(override), f"env override GLM_GPU_LAYERS={override}"
        except ValueError:
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return default_gpu_layers, "NVIDIA GPU detected"
    except Exception:
        pass

    return 0, "GPU unavailable, falling back to CPU"


def load_llm(model_path, n_ctx, verbose=False):
    from llama_cpp import Llama

    gpu_layers, backend_note = detect_gpu_layers()
    attempts = [
        {
            "label": backend_note,
            "kwargs": {"model_path": model_path, "n_ctx": n_ctx, "n_gpu_layers": gpu_layers, "verbose": verbose},
        }
    ]

    if gpu_layers == 0:
        attempts.append(
            {
                "label": "CPU fallback without mmap",
                "kwargs": {
                    "model_path": model_path,
                    "n_ctx": n_ctx,
                    "n_gpu_layers": 0,
                    "use_mmap": False,
                    "verbose": verbose,
                },
            }
        )

    last_error = None
    for attempt in attempts:
        try:
            llm = Llama(**attempt["kwargs"])
            return llm, attempt["label"]
        except Exception as e:
            last_error = e

    size_gb = os.path.getsize(model_path) / (1024 ** 3)
    raise RuntimeError(
        "Model load failed after retry.\n"
        f"path: {model_path}\n"
        f"size_gb: {size_gb:.1f}\n"
        f"last_error: {last_error}\n"
        "Try rerunning `./glm --install`, ensure enough RAM is free, or set GLM_GPU_LAYERS=0 explicitly."
    )
