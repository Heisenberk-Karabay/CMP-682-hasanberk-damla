#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${ENV_PATH:-"$SCRIPT_DIR/.venv"}"
LIGHTGLUE_DIR="${LIGHTGLUE_DIR:-"$SCRIPT_DIR/LightGlue"}"
PYTHON_BIN="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
FEATURE_TYPE="${FEATURE_TYPE:-superpoint}"
DEVICE="${DEVICE:-}"
MAX_KEYPOINTS="${MAX_KEYPOINTS:-2048}"
RESIZE="${RESIZE:-1600}"
OUTPUT="${OUTPUT:-"$SCRIPT_DIR/lightglue_results.json"}"
VIZ_DIR="${VIZ_DIR:-"$SCRIPT_DIR/match_images"}"
VIZ_N="${VIZ_N:-5}"
SKIP_TORCH_INSTALL=0
FORCE_RECREATE=0
SKIP_EVALUATION=0

usage() {
    cat <<'EOF'
Usage: bash setup_lightglue_env.sh [options]

Options:
  --env-path PATH          Virtual environment path. Default: ./.venv
  --lightglue-dir PATH     LightGlue package path. Default: ./LightGlue
  --python PATH            Python executable to create the venv. Default: python3
  --torch-index-url URL    Optional PyTorch package index URL.
  --feature-type NAME      superpoint, disk, or sift. Default: superpoint
  --device DEVICE          cuda or cpu. Default: auto-detect in evaluator.
  --max-keypoints N        Max keypoints per image. Default: 2048
  --resize N               Resize long edge before matching. Default: 1600
  --output PATH            Results JSON path. Default: ./lightglue_results.json
  --viz-dir PATH           Match image output directory. Default: ./match_images
  --viz-n N                Number of random pairs to visualize. Default: 5
  --skip-torch-install     Do not install torch/torchvision.
  --skip-evaluation        Set up the environment without running evaluation.
  --force-recreate         Delete and recreate the virtual environment.
  -h, --help               Show this help.

Environment variables:
  ENV_PATH, LIGHTGLUE_DIR, PYTHON, TORCH_INDEX_URL,
  FEATURE_TYPE, DEVICE, MAX_KEYPOINTS, RESIZE, OUTPUT, VIZ_DIR, VIZ_N
EOF
}

require_value() {
    if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-path)
            require_value "$@"
            ENV_PATH="$2"
            shift 2
            ;;
        --lightglue-dir)
            require_value "$@"
            LIGHTGLUE_DIR="$2"
            shift 2
            ;;
        --python)
            require_value "$@"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --torch-index-url)
            require_value "$@"
            TORCH_INDEX_URL="$2"
            shift 2
            ;;
        --feature-type)
            require_value "$@"
            FEATURE_TYPE="$2"
            shift 2
            ;;
        --device)
            require_value "$@"
            DEVICE="$2"
            shift 2
            ;;
        --max-keypoints)
            require_value "$@"
            MAX_KEYPOINTS="$2"
            shift 2
            ;;
        --resize)
            require_value "$@"
            RESIZE="$2"
            shift 2
            ;;
        --output)
            require_value "$@"
            OUTPUT="$2"
            shift 2
            ;;
        --viz-dir)
            require_value "$@"
            VIZ_DIR="$2"
            shift 2
            ;;
        --viz-n)
            require_value "$@"
            VIZ_N="$2"
            shift 2
            ;;
        --skip-torch-install)
            SKIP_TORCH_INSTALL=1
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=1
            shift
            ;;
        --force-recreate)
            FORCE_RECREATE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if [[ "${PYTHON:-}" == "" ]] && command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Python executable not found: $PYTHON_BIN" >&2
        exit 1
    fi
fi

run() {
    echo ">> $*"
    "$@"
}

python_import_exists() {
    local python_exe="$1"
    local module_name="$2"

    "$python_exe" - "$module_name" <<'PY'
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

if [[ "$ENV_PATH" != /* ]]; then
    ENV_PATH="$SCRIPT_DIR/$ENV_PATH"
fi

if [[ "$LIGHTGLUE_DIR" != /* ]]; then
    LIGHTGLUE_DIR="$SCRIPT_DIR/$LIGHTGLUE_DIR"
fi

if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$SCRIPT_DIR/$OUTPUT"
fi

if [[ "$VIZ_DIR" != /* ]]; then
    VIZ_DIR="$SCRIPT_DIR/$VIZ_DIR"
fi

VENV_PYTHON="$ENV_PATH/bin/python"
VENV_ACTIVATE="$ENV_PATH/bin/activate"

if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]]; then
    VENV_PYTHON="$ENV_PATH/Scripts/python.exe"
    VENV_ACTIVATE="$ENV_PATH/Scripts/activate"
fi

echo "Project folder:   $SCRIPT_DIR"
echo "Environment path: $ENV_PATH"
echo "LightGlue path:   $LIGHTGLUE_DIR"

if [[ "$FORCE_RECREATE" -eq 1 && -d "$ENV_PATH" ]]; then
    echo "Removing existing virtual environment because --force-recreate was provided."
    rm -rf -- "$ENV_PATH"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    mkdir -p -- "$(dirname -- "$ENV_PATH")"
    run "$PYTHON_BIN" -m venv "$ENV_PATH"
else
    echo "Virtual environment already exists; reusing it."
fi

run "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_TORCH_INSTALL" -eq 0 ]]; then
    if python_import_exists "$VENV_PYTHON" torch; then
        echo "torch is already installed in the environment."
    elif [[ -n "$TORCH_INDEX_URL" ]]; then
        run "$VENV_PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
    else
        run "$VENV_PYTHON" -m pip install torch torchvision
    fi
fi

run "$VENV_PYTHON" -m pip install numpy opencv-python tqdm

if [[ ! -d "$LIGHTGLUE_DIR" ]]; then
    echo "LightGlue folder not found at $LIGHTGLUE_DIR. Ship the LightGlue folder with this project before running setup." >&2
    exit 1
elif [[ ! -f "$LIGHTGLUE_DIR/pyproject.toml" ]]; then
    echo "LightGlueDir exists but does not look like the LightGlue Python package: $LIGHTGLUE_DIR" >&2
    exit 1
else
    echo "Using bundled LightGlue package."
fi

(
    cd "$LIGHTGLUE_DIR"
    run "$VENV_PYTHON" -m pip install -e .
)

run "$VENV_PYTHON" -c "import cv2, numpy, torch, tqdm; from lightglue import LightGlue, SuperPoint; print('LightGlue environment is ready.')"

if [[ "$SKIP_EVALUATION" -eq 0 ]]; then
    eval_args=(
        "$SCRIPT_DIR/evaluate_lightglue.py"
        --feature_type "$FEATURE_TYPE"
        --max_keypoints "$MAX_KEYPOINTS"
        --resize "$RESIZE"
        --output "$OUTPUT"
        --viz_dir "$VIZ_DIR"
        --viz_n "$VIZ_N"
    )

    if [[ -n "$DEVICE" ]]; then
        eval_args+=(--device "$DEVICE")
    fi

    kitti_root="$SCRIPT_DIR/kitti/dataset"
    if [[ -d "$kitti_root" ]]; then
        eval_args+=(
            --kitti_root "$kitti_root"
            --kitti_seqs 00
            --kitti_step 5
            --kitti_max_pairs 300
        )
    fi

    fourseasons_root="$SCRIPT_DIR/4seasons"
    fourseasons_recording="recording_2020-03-26_13-32-55"
    if [[ -d "$fourseasons_root/$fourseasons_recording" ]]; then
        eval_args+=(
            --fourseasons_root "$fourseasons_root"
            --fourseasons_poses_root "$fourseasons_root"
            --fourseasons_recordings "$fourseasons_recording"
            --fourseasons_step 5
            --fourseasons_max_pairs 300
        )
    fi

    if [[ "${#eval_args[@]}" -le 9 ]]; then
        echo "No bundled datasets found; skipping evaluation."
    else
        run "$VENV_PYTHON" "${eval_args[@]}"
    fi
fi

cat <<EOF

Activate it with:
  source "$VENV_ACTIVATE"

Results file:
  "$OUTPUT"

Match images:
  "$VIZ_DIR"
EOF
