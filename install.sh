#!/usr/bin/env bash
# ============================================================================
# SimuScene environment setup
#
# Usage:
#   bash install.sh              # create a Python 3.11 venv in ./.venv
#   bash install.sh --conda      # create a conda env named "simuscene"
#   bash install.sh --no-system  # skip apt system packages (ffmpeg/libgl/...)
#
# After it finishes:
#   venv:  source .venv/bin/activate
#   conda: conda activate simuscene
# ============================================================================
set -euo pipefail

USE_CONDA=0
INSTALL_SYSTEM=1
ENV_NAME="simuscene"
PY_VERSION="3.11"

for arg in "$@"; do
  case "$arg" in
    --conda)      USE_CONDA=1 ;;
    --no-system)  INSTALL_SYSTEM=0 ;;
    -h|--help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. System packages (OpenCV video backend + headless graphics libs)
# ---------------------------------------------------------------------------
if [ "$INSTALL_SYSTEM" -eq 1 ]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo ">> Installing system packages (ffmpeg, libgl1, libglib2.0-0) ..."
    SUDO=""
    [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
    $SUDO apt-get update
    $SUDO apt-get install -y ffmpeg libgl1 libglib2.0-0
  else
    echo ">> apt-get not found; skipping system packages."
    echo "   Ensure ffmpeg + OpenGL/glib libraries are available for OpenCV."
  fi
else
  echo ">> --no-system: skipping system packages."
fi

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------
if [ "$USE_CONDA" -eq 1 ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: --conda requested but 'conda' is not on PATH." >&2
    exit 1
  fi
  echo ">> Creating conda env '$ENV_NAME' (python $PY_VERSION) ..."
  conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
  # run pip inside the env without needing 'conda activate' in a script
  PIP=(conda run -n "$ENV_NAME" python -m pip)
else
  echo ">> Creating venv at ./.venv (python $PY_VERSION) ..."
  if command -v "python$PY_VERSION" >/dev/null 2>&1; then
    PYBIN="python$PY_VERSION"
  else
    echo "   python$PY_VERSION not found; falling back to 'python3'."
    PYBIN="python3"
  fi
  "$PYBIN" -m venv .venv
  PIP=(.venv/bin/python -m pip)
fi

# ---------------------------------------------------------------------------
# 3. Python packages
# ---------------------------------------------------------------------------
echo ">> Upgrading pip and installing Python dependencies ..."
"${PIP[@]}" install --upgrade pip
"${PIP[@]}" install \
    openai \
    pandas \
    numpy \
    tqdm \
    opencv-python \
    psutil \
    pillow \
    datasets \
    qwen-vl-utils

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SimuScene environment ready."
if [ "$USE_CONDA" -eq 1 ]; then
  echo "Activate with:  conda activate $ENV_NAME"
else
  echo "Activate with:  source .venv/bin/activate"
fi
echo ""
echo "Next: set your OpenAI key for data_construction steps 1-2:"
echo "  export OPENAI_API_KEY=\"sk-...\""
echo "and configure your local vLLM/SGLang servers for the"
echo "DeepSeek-R1-0528 / Qwen2.5-VL-72B steps (see ReadMe.txt section 3)."
echo "============================================================"
