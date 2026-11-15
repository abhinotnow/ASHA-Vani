# Hosting Gnani Evon v3.3 (`gnani-evon-v3.3-30B-A3B`) for Free on Kaggle `2× T4`

ASHA-Vani supports **Gnani Evon v3.3** (`gnani/gnani-evon-v3.3-30B-A3B`, hybrid Mamba2-Transformer MoE, `nemotron_h_moe` architecture) as the primary extraction LLM via any OpenAI-compatible `/v1/chat/completions` endpoint, with automatic failover to **Groq (`llama-3.3-70b-versatile` → `llama-3.1-8b-instant`)** and deterministic regex extraction (`_fallback_extract`).

Because Kaggle's free **GPU T4 × 2** accelerator provides **32 GB total VRAM** (`16 GB` per GPU), the **17.1 GB `IQ4_XS` GGUF** quantization from [`luminenio/gnani-evon-v3.3-30B-A3B-GGUF`](https://huggingface.co/luminenio/gnani-evon-v3.3-30B-A3B-GGUF) fits completely on GPU with ~14.9 GB headroom for context and KV cache.

> [!IMPORTANT]
> Always download the 17.1 GB GGUF weights into `/tmp/evon` rather than `/kaggle/working`. Kaggle enforces a ~20 GB quota on `/kaggle/working`, whereas `/tmp` has ample scratch space for the model weights and build tree, leaving `/kaggle/working` free to cache the compiled `llama-server` binary.

---

## Step 1: Create a Kaggle Notebook

1. Open [Kaggle](https://www.kaggle.com/) (phone-verified account, no credit card required).
2. Create a new notebook and open **Notebook Settings** (right sidebar):
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** `On`

---

## Step 2: Cell 1 — Download `IQ4_XS` GGUF (`17.1 GB`) to `/tmp/evon`

Run this cell first so model weights are stored in `/tmp/evon` (keeping `/kaggle/working` under its 20 GB limit):

```bash
%%bash
set -euo pipefail

mkdir -p /tmp/evon
pip install -q --upgrade huggingface_hub
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="luminenio/gnani-evon-v3.3-30B-A3B-GGUF",
    allow_patterns=["*IQ4_XS*.gguf", "*iq4_xs*.gguf"],
    local_dir="/tmp/evon",
)
PY
ls -lh /tmp/evon
```

---

## Step 3: Cell 2 — Build CUDA `llama-server` (`sm_75`) & Download `cloudflared`

Because official `ggml-org/llama.cpp` Ubuntu release tarballs are CPU-only, build `llama-server` directly for Kaggle's T4 GPUs (`sm_75`) and cache the binary in `/kaggle/working/llama-server`:

```bash
%%bash
set -euo pipefail

mkdir -p /tmp/llama-bin

if [ -x /kaggle/working/llama-server ]; then
  echo "Reusing cached /kaggle/working/llama-server"
  cp /kaggle/working/llama-server /tmp/llama-bin/llama-server
else
  if [ ! -d /tmp/llama-src ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama-src
  fi
  rm -rf /tmp/llama-src/build
  cmake -S /tmp/llama-src -B /tmp/llama-src/build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_NO_VMM=ON \
    -DGGML_CUDA_FA=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF
  cmake --build /tmp/llama-src/build --target llama-server -j4
  cp /tmp/llama-src/build/bin/llama-server /tmp/llama-bin/llama-server
  cp /tmp/llama-src/build/bin/llama-server /kaggle/working/llama-server
fi

# Download Cloudflare Tunnel binary (free public HTTPS URL, no account needed)
curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /tmp/cloudflared
chmod +x /tmp/cloudflared
echo "Setup complete!"
```

> **Why these CMake flags are needed on Kaggle `2× T4`:**
> - `-DGGML_CUDA_NO_VMM=ON` bypasses the missing `CUDA::cuda_driver` stub link target on Kaggle's read-only `/usr/local/cuda/lib64/stubs` directory.
> - `-DGGML_CUDA_FA=OFF` skips compiling 200+ FlashAttention `.cu` kernels across Kaggle's 4 CPU cores, cutting build time dramatically.
> - `-DCMAKE_CUDA_ARCHITECTURES=75` compiles specifically for NVIDIA Turing (`T4` / `sm_75`).
> - Caching to `/kaggle/working/llama-server` makes subsequent notebook sessions start in seconds.

---

## Step 4: Cell 3 — Launch `llama-server` Split Across `2× T4` + `cloudflared` Tunnel

```python
import glob, os, re, subprocess, time, urllib.request

gguf_files = sorted(glob.glob("/tmp/evon/**/*IQ4_XS*.gguf", recursive=True))
assert gguf_files, "No IQ4_XS GGUF file found in /tmp/evon"
model_path = gguf_files[0]

server_bin = "/tmp/llama-bin/llama-server"
assert os.path.exists(server_bin), "llama-server binary not found in /tmp/llama-bin"
os.chmod(server_bin, 0o755)

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{os.path.dirname(server_bin)}:{env.get('LD_LIBRARY_PATH', '')}"

server_proc = subprocess.Popen(
    [
        server_bin,
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", "8080",
        "-ngl", "99",
        "--tensor-split", "1,1",
        "-c", "4096",
        "--parallel", "2",
        "--alias", "gnani/gnani-evon-v3.3-30B-A3B",
    ],
    env=env,
    stdout=open("/tmp/llama-server.log", "w"),
    stderr=subprocess.STDOUT,
)

print("Waiting for llama-server on :8080 (loading 17.1 GB model across 2x T4)...")
ready = False
for _ in range(180):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as r:
            if r.status == 200:
                ready = True
                print("llama-server is ready!")
                break
    except Exception:
        time.sleep(2)
assert ready, "llama-server did not become healthy in time; inspect /tmp/llama-server.log"

tunnel_proc = subprocess.Popen(
    ["/tmp/cloudflared", "tunnel", "--url", "http://127.0.0.1:8080"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in tunnel_proc.stdout:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        print("========================================================")
        print("PASTE THIS INTO YOUR LOCAL .env:")
        print(f"EVON_BASE_URL={m.group(0)}/v1")
        print("========================================================")
        break
```

---

## Step 5: Configure Local ASHA-Vani `.env` & Verify

Copy the printed URL into your local `.env`:

```env
EVON_BASE_URL=https://<random-subdomain>.trycloudflare.com/v1
EVON_API_KEY=EMPTY
EVON_MODEL=gnani/gnani-evon-v3.3-30B-A3B
EVON_TIMEOUT_S=8.0
```

Start the server and verify:

```bash
uv run asha-vani
```

Check `http://127.0.0.1:8000/health`:
- `"evon_configured": true`
- `"evon_model": "gnani/gnani-evon-v3.3-30B-A3B"`

If the Kaggle session stops or `EVON_TIMEOUT_S` (`8.0s`) expires, `extract_turn()` in `src/asha_vani/extraction.py` automatically falls back to Groq (`GROQ_API_KEY`) and then local regex extraction (`_fallback_extract`).
