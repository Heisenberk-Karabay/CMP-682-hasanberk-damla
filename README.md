# CMP-682: LightGlue Evaluation

Evaluates the [LightGlue](https://github.com/cvg/LightGlue) feature matcher on the **KITTI Odometry** and **4Seasons** datasets, reporting epipolar precision, pose AUC, and match timing.

## Running the setup script

The single entry point is `setup_lightglue_env.sh`. It creates a Python virtual environment, installs all dependencies, and runs the evaluation automatically.

**Linux**

```bash
bash setup_lightglue_env.sh
```

**Windows:** open **Git Bash** and run the same command:

```bash
bash setup_lightglue_env.sh
```

> Git Bash ships with Git for Windows and provides the POSIX shell environment required by the script. The Windows Command Prompt and PowerShell are not supported.

### Useful options

| Flag | Default | Description |
|---|---|---|
| `--feature-type` | `superpoint` | Feature extractor: `superpoint`, `disk`, or `sift` |
| `--device` | auto | `cuda` or `cpu` |
| `--output` | `./lightglue_results.json` | Results file path |
| `--viz-dir` | `./match_images` | Match visualization output folder |
| `--skip-evaluation` | - | Set up the environment only, skip evaluation |
| `--force-recreate` | - | Delete and recreate the virtual environment |

Run `bash setup_lightglue_env.sh --help` for the full list.

## Outputs

**`lightglue_results.json`:** evaluation metrics for each dataset/sequence in JSON format. Fields per entry:

- `avg_matches`, `avg_precision_pct`, `avg_inliers`: matching quality
- `AUC@5deg`, `AUC@10deg`, `AUC@20deg`: pose estimation accuracy
- `avg_R_error_deg`, `avg_t_error_deg`: mean rotation and translation errors
- `avg_match_time_ms`: matcher throughput

**`match_images/`:** side-by-side visualizations of a random sample of matched image pairs (5 per dataset by default), useful for a quick qualitative check.

## Dataset

The bundled data is a **small subset** of the full benchmarks:

- **4Seasons:** a single short recording (`recording_2020-03-26_13-32-55`, ~100 frames from `cam0`)
- **KITTI:** sequence 00 is used if the full dataset is placed under `kitti/dataset/`

The full KITTI and 4Seasons datasets are much larger and are not included in this repository. The included subset is sufficient to reproduce the numbers in `lightglue_results.json`.

Since the included subset is small, **you may observe lower metric values than those reported**; to reproduce the exact numbers, download the full datasets and run the script against them.

## Device / GPU

The evaluation script **auto-detects** the available device: it runs on **GPU** when CUDA is available and falls back to **CPU** otherwise.

This demo runs on **CPU** because we cannot assume your system has a compatible GPU. With the appropriate CUDA drivers and a CUDA-enabled PyTorch build installed, it will run on GPU automatically, or you can force it with `--device cuda`.

To install PyTorch with CUDA support, follow the instructions at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and pass the resulting index URL to the setup script:

```bash
bash setup_lightglue_env.sh --torch-index-url https://download.pytorch.org/whl/cu121
```
