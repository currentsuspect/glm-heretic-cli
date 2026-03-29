import os
import sys
from huggingface_hub import hf_hub_download

from .runtime import detect_gpu_layers

REPO = "DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF"
FILE = "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q4_K_M.gguf"


def print_header():
    print("GLM install")
    print("model: GLM-4.7-Flash Heretic")
    print(f"repo: {REPO}")
    print()


def ensure_model():
    print("Caching model...")
    model_path = hf_hub_download(repo_id=REPO, filename=FILE)
    size_mb = os.path.getsize(model_path) // (1024 * 1024)
    print(f"cached: {model_path}")
    print(f"size:   {size_mb} MB")
    print()
    return model_path


def print_backend():
    gpu_layers, backend_note = detect_gpu_layers()
    print(f"backend: {backend_note}")
    print(f"n_gpu_layers: {gpu_layers}")
    print()


def handoff(args):
    if not args:
        os.execvp("python3", ["python3", "-m", "glm_cli.chat"])

    mode = args[0]
    if mode == "--agent":
        os.execvp("python3", ["python3", "-m", "glm_cli.agent", *args[1:]])
    if mode == "--serve":
        os.execvp("python3", ["python3", "-m", "glm_cli.server", *args[1:]])
    os.execvp("python3", ["python3", "-m", "glm_cli.chat", *args])


def main():
    args = sys.argv[1:]
    if args and args[0] == "--help":
        print(
            "Usage:\n"
            "  glm --install\n"
            "  glm --install --agent\n"
            "  glm --install --serve\n"
            '  glm --install "your prompt"\n'
        )
        return

    print_header()
    ensure_model()
    print_backend()
    print("Launching GLM...")
    print()
    handoff(args)


if __name__ == "__main__":
    main()
