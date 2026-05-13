# HARDWARE.md — reproducible GPU setup

InferBench's GPU benchmarks target NVIDIA RTX A2000-class cards (A2000
12GB Ampere and A2000 ADA 16GB are both validated). This doc is the
runbook: every command needed to take a stock box from blank to
"`make bench-gpu SCENARIO=B`" producing valid results.

Two paths covered:

- **§1–§8**: Ubuntu 24.04 with system-installed CUDA + cuDNN + TensorRT
  (admin required for the CUDA toolkit install).
- **§9**: **Windows / locked-down boxes** — pure pip user-space install,
  no admin needed. Validated on Windows 11, driver 581.95, Python 3.12.

If your box has a different Ada/Ampere GPU (A10, RTX 4000 SFF, RTX 4090,
L4), the same instructions apply — the version pins below are CUDA 12.x
+ cuDNN 9.x + TensorRT 10.x and work across the whole sm_8x / sm_9x
family.

---

## 1. Target software versions (Linux path)

| Component         | Version       | Why this pin                                                  |
|-------------------|---------------|---------------------------------------------------------------|
| NVIDIA driver     | ≥ 550         | Minimum that ships CUDA 12.4 runtime; A2000 ADA is sm_89      |
| CUDA Toolkit      | 12.4 or 12.6  | onnxruntime-gpu 1.22+ is built against CUDA 12.x              |
| cuDNN             | 9.x           | onnxruntime-gpu 1.22+ links cuDNN 9; do **not** mix with 8.x  |
| TensorRT          | 10.4 or 10.6  | TensorRT 10 is CUDA 12 compatible; needed for the TRT EP      |
| onnxruntime-gpu   | ≥ 1.22        | Adds `preload_dlls()` API used on Windows path                |
| Python            | 3.11 or 3.12  | Matches the existing `.venv` and CI                           |

Pinning rationale: ORT EPs (CUDA, TensorRT) require *exact* CUDA/cuDNN
ABI alignment. A version mismatch silently falls back to
`CPUExecutionProvider` with no warning louder than a slow benchmark.
We verify the EP is actually loaded with the smoke test in §5.

---

## 2. Verify the GPU

```bash
nvidia-smi
# Expected: A2000 ADA or A2000 12GB, driver ≥ 550, CUDA 12.x or 13.x.

lspci | grep -i nvidia
# Expected: at least one NVIDIA Corporation device.
```

If `nvidia-smi` is missing, install the proprietary driver first:

```bash
sudo ubuntu-drivers install nvidia:550
sudo reboot
```

---

## 3. Install CUDA + cuDNN + TensorRT (Linux)

NVIDIA's apt repo is the most reproducible path. Skip the .run installers.

```bash
# Add NVIDIA CUDA apt repo (Ubuntu 24.04 / noble)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Toolkit + cuDNN + TensorRT
sudo apt install -y \
    cuda-toolkit-12-6 \
    cudnn9-cuda-12 \
    libnvinfer10 libnvinfer-plugin10 libnvinfer-dispatch10 \
    libnvonnxparsers10 libnvinfer-headers-dev tensorrt
```

Add CUDA to `PATH` / `LD_LIBRARY_PATH` (append to `~/.bashrc`):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

Verify:

```bash
nvcc --version          # CUDA 12.6
dpkg -l | grep cudnn9   # cudnn9-cuda-12 installed
dpkg -l | grep nvinfer  # TensorRT 10.x installed
```

---

## 4. Python deps (Linux)

```bash
make venv
make install
make gpu-install   # uninstalls cpu onnxruntime, installs onnxruntime-gpu
```

`onnxruntime` and `onnxruntime-gpu` are mutually exclusive — installing
both leaves you with whichever loaded last and silent EP fallbacks. The
`gpu-install` target handles this in the correct order.

---

## 5. Smoke test the EP

```bash
make gpu-smoke
```

Expected output:

```
ort version           : 1.22.0
available providers   : ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
session providers     : ['CUDAExecutionProvider', 'CPUExecutionProvider']
effective backend tag : onnxruntime-cuda
sample inference ok   : POSITIVE (score 0.9998)
GPU SMOKE TEST: PASS
```

If `available providers` doesn't list `CUDAExecutionProvider`, your
onnxruntime-gpu can't see CUDA — go back to §3 (almost always a
`LD_LIBRARY_PATH` or cuDNN version mismatch).

---

## 6. Export the fp16 model

```bash
make export-model         # fp32 ONNX (~260 MB)
make export-model-fp16    # fp16 graph-transform (~130 MB), no GPU needed
make quantize-model       # int8 dynamic quant (~67 MB)
```

---

## 7. Run the GPU benchmark

```bash
# Serve with the GPU profile
make serve-gpu

# In a second shell, run the concurrency sweep against the GPU server
make bench-gpu SCENARIO=B
```

Results land in `results/run_NNN/`. The scenario snapshot records the
effective backend tag (and GPU name/driver via nvidia-smi) so any reader
can verify the run was actually on CUDA, not silently on CPU.

---

## 8. Re-running on Lambda Cloud / vast.ai (fallback)

If the A2000 queue is blocked, an A10G (sm_86) on Lambda Cloud is the
closest commodity match (~$0.60/hr). Same instructions §3–§7 apply. Tag
the run with `make bench-gpu HARDWARE=a10g-lambda` so it doesn't get
mixed up with A2000 numbers in the headline charts.

---

## 9. Windows setup (no admin, pure pip)

This path was validated on a locked-down Windows 11 box with the
NVIDIA RTX A2000 12GB (Ampere, sm_86), driver 581.95 (CUDA 13.0 runtime),
and Python 3.12. The trick: pip-installed `torch` with the CUDA index ships
all the CUDA DLLs (cublas, cudnn, cufft, cudart, etc.) in `torch/lib/`,
and `onnxruntime.preload_dlls()` (ORT 1.21+) discovers them at import time.
`inferbench.engine.model_runner` calls `preload_dlls()` automatically on
Windows, so all this is wired up out of the box.

There is no Makefile on Windows — run the commands by hand in PowerShell.

### 9.1 Verify the driver

```powershell
nvidia-smi
# Expected: NVIDIA RTX A2000 (any model), driver version, CUDA version line.
python --version
# Expected: Python 3.11.x or 3.12.x
```

### 9.2 Clone + venv

```powershell
git clone https://github.com/aarmens702-hub/Inference-Benchmarks.git
cd Inference-Benchmarks
git checkout claude/setup-rtx-2000-mUkSz   # or main once merged
python -m venv .venv
```

### 9.3 Install Python deps

Note: use full paths to `.\.venv\Scripts\...` instead of activating, so
you don't fight PowerShell's `Set-ExecutionPolicy`. Each command can be
run one at a time — PowerShell 5.x doesn't support `&&`.

```powershell
.\.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\pip uninstall -y onnxruntime
.\.venv\Scripts\pip install "onnxruntime-gpu>=1.22" onnxconverter-common
```

### 9.4 Install the CUDA-enabled torch wheel (critical)

The default `torch` from PyPI is CPU-only on Windows and ships no CUDA
DLLs. We need the cu124 wheel:

```powershell
.\.venv\Scripts\pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

Verify it sees the GPU:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no')"
# Expected: True NVIDIA RTX A2000 ...
```

### 9.5 Smoke test

```powershell
.\.venv\Scripts\python -m inferbench.models.export_model
.\.venv\Scripts\python -m inferbench.tools.gpu_smoke
```

Last line must be `GPU SMOKE TEST: PASS`.

### 9.6 Run the benchmark

```powershell
# Shell 1: serve
$env:INFERBENCH_CONFIG = "configs/server-gpu.yaml"
.\.venv\Scripts\uvicorn inferbench.server.app:app --host 127.0.0.1 --port 8000

# Shell 2: bench
.\.venv\Scripts\python -m inferbench.benchmarks.run_benchmark --scenario B --hardware a2000-12gb-windows
```

Results land in `results/run_NNN/`.
