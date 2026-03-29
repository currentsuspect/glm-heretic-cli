from huggingface_hub import hf_hub_download
import time
from .runtime import load_llm

REPO = ""
FILE = "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q4_K_M.gguf"

print(f"Downloading {FILE} from {REPO}...")
model_path = hf_hub_download(repo_id=REPO, filename=FILE)
print(f"Model downloaded to: {model_path}")

print("\nLoading model backend...")
llm, backend_note = load_llm(model_path=model_path, n_ctx=4096, verbose=True)
print(backend_note)

print("\n--- Model loaded! Testing inference ---\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in 3 sentences."},
]

start = time.time()
output = llm.create_chat_completion(
    messages=messages,
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
elapsed = time.time() - start

response = output["choices"][0]["message"]["content"]
tokens = output["usage"]["completion_tokens"]

print(f"Response:\n{response}\n")
print(f"Time: {elapsed:.1f}s | Tokens: {tokens} | Speed: {tokens/elapsed:.1f} tok/s")
