# HARDWARE.md — reproducible GPU setup

InferBench's GPU benchmarks target an **NVIDIA RTX A2000 ADA (16 GB)**.
This doc is the runbook: every command needed to take a stock Ubuntu
24.04 box from blank to "`make bench-gpu SCENARIO=B`" producing valid
results.

If your box has a different Ada/Ampere GPU (A10, RTX 4000 SFF, RTX 4090,
L4), the same instructions apply — the version pins below are CUDA 12.x
+ cuDNN 9.x + TensorRT 10.x and work across the whole sm_8x / sm_9x
family.

---

## 1. Target software versions

| Component         | Version       | Why this pin                                                  |
|-------------------|---------------|---------------------------------------------------------------|
| NVIDIA driver     | ≥ 550         | Minimum that ships CUDA 12.4 runtime; A2000 ADA is sm_89      |
| CUDA Toolkit      | 12.4 or 12.6  | onnxruntime-gpu 1.20.x is built against CUDA 12.x             |
| cuDNN             | 9.x           | onnxruntime-gpu 1.20.x links cuDNN 9; do **not** mix with 8.x |
| TensorRT          | 10.4 or 10.6  | TensorRT 10 is CUDA 12 compatible; needed for the TRT EP      |
| onnxruntime-gpu   | 1.20.1        | First release with first-class CUDA-12 + cuDNN-9 wheels       |
| Python            | 3.11          | Matches the existing `.venv` and CI                           |

Pinning rationale: ORT EPs (CUDA, TensorRT) require *exact* CUDA/cuDNN
ABI alignment. A version mismatch silently falls back to
`CPUExecutionProvider` with no warning louder than a slow benchmark.
We verify the EP is actually loaded with the smoke test in §5.

---

## 2. Verify the GPU

```bash
nvidia-smi
# Expected: A2000 ADA, driver ≥ 550, CUDA 12.x in the top-right corner.

lspci | grep -i nvidia
# Expected: at least one NVIDIA Corporation device.
```

If `nvidia-smi` is missing, install the proprietary driver first:

```bash
sudo ubuntu-drivers install nvidia:550
sudo reboot
```

---

## 3. Install CUDA + cuDNN + TensorRT

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

## 4. Python deps

```bash
make venv
make install

# GPU-specific wheel (not in requirements.txt because it's optional and
# conflicts with the CPU onnxruntime wheel installed by `make install`):
.venv/bin/pip uninstall -y onnxruntime
.venv/bin/pip install onnxruntime-gpu==1.20.1

# onnxconverter-common is needed by the fp16 export path:
.venv/bin/pip install onnxconverter-common
```

`onnxruntime` and `onnxruntime-gpu` are mutually exclusive — installing
both leaves you with whichever loaded last and silent EP fallbacks. The
uninstall step above is mandatory.

---

## 5. Smoke test the EP

```bash
.venv/bin/python -m inferbench.tools.gpu_smoke
```

Expected output:

```
ort version           : 1.20.1
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
# fp32 export still works on GPU box (uses torch + optimum on import)
make export-model

# fp16 conversion (graph transform, no GPU needed at export time)
make export-model-fp16
```

You should now have:

```
models/distilbert-sst2-fp32/  (~260 MB)
models/distilbert-sst2-fp16/  (~130 MB)
models/distilbert-sst2-int8/  (~67 MB)
```

---

## 7. Run the GPU benchmark

```bash
# Serve with the GPU profile
INFERBENCH_CONFIG=configs/server-gpu.yaml make serve

# In a second shell, run the concurrency sweep against the GPU server
make bench-gpu SCENARIO=B
```

Results land in `results/run_NNN/`. The scenario snapshot
(`scenario.yaml`) records the effective backend tag so any reader can
verify the run was actually on CUDA, not silently on CPU.

---

## 8. Re-running on Lambda Cloud / vast.ai (fallback)

If the A2000 ADA queue is blocked, an A10G (sm_86) on Lambda Cloud is
the closest commodity match (~$0.60/hr). Same instructions §3–§7 apply.
Tag the run in `scenario.yaml` with `hardware: a10g` so it doesn't get
mixed up with A2000 numbers in the headline charts.
