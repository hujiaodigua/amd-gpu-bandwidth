# amd-gpu-bandwidth

A command-line tool for measuring GPU **memory bandwidth** and **PCIe transfer bandwidth**, written in plain C with no third-party dependencies.

| Tool | Backend | Notes |
|---|---|---|
| `vk_bandwidth` | Vulkan | **Recommended**. Works on AMD (incl. RDNA1/GCN), NVIDIA, Intel |
| `bandwidth_test` | OpenCL | Only runs on cards with a working OpenCL runtime (see Dependencies) |

---

## How It Works

### 1. Real memory traffic via `vkCmdCopyBuffer`

Vulkan has no "measure bandwidth" API. The idea is simple: **make the GPU's DMA copy engine move a large amount of data, then measure the throughput with a clock**.

`vkCmdCopyBuffer` is recorded in a command buffer and executed by the GPU's **DMA engine (copy engine / SDMA)** after submission — it does not go through shaders or the CPU, so it measures pure hardware copy throughput.

### 2. Two memory types distinguish "within VRAM" from "across PCIe"

Vulkan classifies memory by properties. This tool uses two kinds:

| Memory type | Physical location | Purpose |
|---|---|---|
| `DEVICE_LOCAL` | On-board VRAM | Traverses the GPU's local memory bus |
| `HOST_VISIBLE` | System RAM (BAR / GTT), CPU-mappable | The GPU must cross PCIe to reach it |

On a discrete GPU, `HOST_VISIBLE` memory physically lives in **system RAM**, so any copy between it and `DEVICE_LOCAL` memory is a **PCIe transfer**.

### 3. Three copies = three bandwidths

```
D2D:  device_local A ──copy──> device_local B     → memory bandwidth (read + write)
H2D:  host_visible  ──copy──> device_local        → PCIe write (host → device)
D2H:  device_local  ──copy──> host_visible        → PCIe read (device → host)
```

- D2D moves each byte twice (read once, write once), so effective bytes are counted as `2 × size`.
- H2D / D2H are one-directional, counted as `1 × size`.

### 4. Timing method

- A command buffer holds **K copies** so each submission moves roughly 512MB, amortizing submit overhead.
- A **warm-up** pass runs first (page allocation, DMA channel steady state).
- Host-side `clock_gettime(CLOCK_MONOTONIC)` times each submission: `t0 → vkQueueSubmit → vkQueueWaitIdle → t1`, repeated for **best / average**.
- Bandwidth = total bytes ÷ elapsed time.

> Why not GPU timestamp queries: RADV's transfer-queue timestamps are unreliable (produce absurd values in practice). Host-side timing with large batches is more robust.

### 5. Interpreting the numbers

- **Memory copy bandwidth** (e.g. 77 GB/s) ≠ theoretical peak (e.g. 112 GB/s). A copy is "read + write" on a half-duplex GDDR bus with read/write `turnaround` overhead, so ~65–75% of peak is normal. Approaching peak requires one-directional read-only / write-only kernels (not included here).
- **PCIe bandwidth** ≈ theoretical link bandwidth × efficiency (usually 85–90%), limited by the negotiated PCIe speed/width (check `LnkSta` via `sudo lspci -vvv`).

---

## Dependencies

### Vulkan (`vk_bandwidth`, recommended)

**Build dependencies** (headers + loader dev):

```bash
sudo apt install libvulkan-dev
```

**Runtime dependencies**:

| Package | Purpose | Required |
|---|---|---|
| `libvulkan1` | Vulkan loader (loads vendor ICDs) | Yes (pulled in by `libvulkan-dev`) |
| `mesa-vulkan-drivers` | RADV driver (AMD) / ANV driver (Intel) | Yes for AMD / Intel GPUs |
| `nvidia-driver-*` | NVIDIA proprietary driver, ships its own ICD | Yes for NVIDIA GPUs |
| `vulkan-tools` | Provides `vulkaninfo` for verification | Optional, recommended |

One-shot install by GPU vendor:

```bash
# AMD / Intel GPU
sudo apt install libvulkan-dev mesa-vulkan-drivers vulkan-tools

# NVIDIA GPU
sudo apt install libvulkan-dev vulkan-tools nvidia-driver-580
```

Verify your GPU is visible:

```bash
vulkaninfo --summary | grep -iE "deviceName|driverName"
```

> This tool only uses `vkCmdCopyBuffer` — no shaders — so `glslang-tools` / `shaderc` / SPIR-V are **not** required.

### OpenCL (`bandwidth_test`, optional)

**Build dependencies**:

```bash
sudo apt install ocl-icd-opencl-dev
```

**Runtime dependencies** (an OpenCL ICD, one per vendor):

| GPU | Package | Notes |
|---|---|---|
| NVIDIA | `cuda-opencl-*` (ships with NVIDIA driver) | Works out of the box |
| AMD (ROCm-supported GCN/Vega/CDNA) | `rocm-opencl-runtime` | Requires AMD's repo |
| CPU (testing only) | `pocl-opencl-icd` | Measures the CPU, not a GPU |
| AMD RDNA1 (RX 5300/5500) | none | **Unsupported** — use the Vulkan tool |

---

## Building

```bash
cd amd-gpu-bandwidth

make vk_bandwidth    # build only the Vulkan tool (no OpenCL headers needed)
make                 # build both (requires ocl-icd-opencl-dev)
make clean
```

## Usage

```bash
./vk_bandwidth                  # default 256MB, tests all discrete GPUs
./vk_bandwidth -h               # show help
./vk_bandwidth -l               # list discrete GPUs without benchmarking
./vk_bandwidth -s 512           # set buffer size in MB
./vk_bandwidth -d 5300          # only test devices whose name contains "5300"
./vk_bandwidth -d 5300 -s 64    # combined
```

Example output:

```
== AMD Radeon RX 550 Series (RADV POLARIS11) ==
  driverVersion 23.2.1   device-local heap 4096.0 MB
  buffer size 256 MB

  -- VRAM bandwidth --
  D2D copy         best    77.38 GB/s   avg    76.98 GB/s

  -- PCIe bandwidth --
  H2D (write)      best     5.80 GB/s   avg     5.78 GB/s
  D2H (read)       best     5.14 GB/s   avg     5.12 GB/s
```

---

## Test Results

Measured with `./vk_bandwidth` (256 MB buffer) on an AMD Ryzen 9 5900X host (Ubuntu 22.04):

| GPU | VRAM D2D copy | PCIe H2D (write) | PCIe D2H (read) |
|---|---|---|---|
| AMD Radeon RX 5300 (RADV NAVI14) | **150.23 GB/s** | **14.37 GB/s** | **13.69 GB/s** |
| Tesla P4 | **147.96 GB/s** | **6.66 GB/s** | **6.61 GB/s** |

Full output:

```
== AMD Radeon RX 5300 (RADV NAVI14) ==
  driverVersion 23.2.1   device-local heap 3056.0 MB
  buffer size 256 MB

  -- VRAM bandwidth --
  D2D copy         best   150.23 GB/s   avg   146.10 GB/s

  -- PCIe bandwidth --
  H2D (write)      best    14.37 GB/s   avg    14.31 GB/s
  D2H (read)       best    13.69 GB/s   avg    13.64 GB/s

== Tesla P4 ==
  driverVersion 580.504.576   device-local heap 7680.0 MB
  buffer size 256 MB

  -- VRAM bandwidth --
  D2D copy         best   147.96 GB/s   avg   147.77 GB/s

  -- PCIe bandwidth --
  H2D (write)      best     6.66 GB/s   avg     6.65 GB/s
  D2H (read)       best     6.61 GB/s   avg     6.60 GB/s
```

> The RX 5300's PCIe bandwidth is capped at ~14 GB/s because it sits behind a Navi 10 PCIe switch with a PCIe 4.0 ×8 upstream link; the Tesla P4 runs at PCIe 3.0 ×8.

---

## Notes

- Buffer size is automatically clamped to 1/4 of VRAM to avoid allocation failure.
- Memory results reflect **copy** bandwidth (read + write), not theoretical peak; read/write kernels are needed for peak figures (not included here).
- PCIe results are limited by the actual negotiated link; for cards behind a PCIe switch, the bottleneck may be the switch's upstream port rather than the card itself (check `LnkSta` via `sudo lspci -vvv`).
- All results use decimal GB/s (1 GB = 10⁹ bytes).
