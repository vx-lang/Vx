"""A stand-in for `vllm` in the tests of run_matrix.run_engine: it prints the lines of the
pilot's vLLM 0.31.0 runs and answers /health and /v1/completions. FAKE_VLLM picks what it does:
`serves`, `oom_at_load`, or `dies_on_request`."""

import http.server
import json
import os
import sys

if sys.argv[1:] == ["--version"]:
    print("0.31.0")
    sys.exit(0)

mode = os.environ["FAKE_VLLM"]
port = int(sys.argv[sys.argv.index("--port") + 1])
if mode == "oom_at_load":
    print("Model loading took 135.43 GiB", flush=True)
    print("torch.OutOfMemoryError: CUDA out of memory", file=sys.stderr, flush=True)
    sys.exit(1)
print("GPU KV cache size: 1000 tokens", flush=True)


class Engine(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if mode == "dies_on_request":
            print("EngineDeadError", file=sys.stderr, flush=True)
            os._exit(1)
        body = json.dumps({"choices": [{"text": "token"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


try:
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Engine).serve_forever()
except KeyboardInterrupt:
    pass
