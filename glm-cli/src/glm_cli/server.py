from huggingface_hub import hf_hub_download
import json
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from .runtime import load_llm

REPO = "DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF"
FILE = "GLM-4.7-Flash-Uncen-Hrt-NEO-CODE-MAX-imat-D_AU-Q4_K_M.gguf"

print("Loading model...")
model_path = hf_hub_download(repo_id=REPO, filename=FILE)
llm, backend_note = load_llm(model_path=model_path, n_ctx=4096, verbose=False)
print(f"Model loaded. {backend_note}")

def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            gen = llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, top_p=0.9, stop=["<|endoftext|>"],
                stream=True,
            )
            for chunk in gen:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    delta["content"] = strip_thinking(delta["content"])
                data = json.dumps(chunk)
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            output = llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, top_p=0.9, stop=["<|endoftext|>"],
            )
            output["choices"][0]["message"]["content"] = strip_thinking(
                output["choices"][0]["message"]["content"]
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(output).encode())

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "data": [{"id": "glm-4.7-flash-heretic", "object": "model"}]
            }).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    port = 8080
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"OpenAI-compatible API at http://localhost:{port}/v1")
    print("Endpoints: POST /v1/chat/completions, GET /v1/models")
    server.serve_forever()
