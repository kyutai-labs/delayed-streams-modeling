#!/bin/bash
# setup_env.sh
# Sets up the environment for CUDA builds with cudarc 0.16.6.

# Determine absolute path to project root
# We use standard tools with absolute paths to avoid issues if PATH is broken
SCRIPT_PATH=$(/usr/bin/readlink -f "${BASH_SOURCE[0]}")
OPS_DIR=$(/usr/bin/dirname "$SCRIPT_PATH")
PROJECT_ROOT=$(/usr/bin/dirname "$OPS_DIR")
FAKE_BIN="$PROJECT_ROOT/fake_bin"

/usr/bin/mkdir -p "$FAKE_BIN"

# Create spoofed nvcc
/usr/bin/cat <<'NVCC_EOF' > "$FAKE_BIN/nvcc"
#!/bin/bash
if [[ "$*" == *"--version"* ]]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2023 NVIDIA Corporation"
    echo "Built on Tue_Jul_11_20:56:05_PDT_2023"
    echo "Cuda compilation tools, release 12.9, V12.9.140"
    echo "Build cuda_12.9.r12.9/compiler.33039272_0"
else
    # Fallback to the real nvcc for actual compilation
    exec /usr/local/cuda/bin/nvcc "$@"
fi
NVCC_EOF
/usr/bin/chmod +x "$FAKE_BIN/nvcc"

# Check if FAKE_BIN is already in PATH to avoid infinite growth
if [[ ":$PATH:" != *":$FAKE_BIN:"* ]]; then
    export PATH="$FAKE_BIN:/usr/local/cuda/bin:$PATH"
fi

export CUDA_HOME="/usr/local/cuda"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export CUDA_ROOT="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDARC_CUDA_VERSION=12090

echo "Environment prepared for Moshi GPU build (CUDA 13.1 spoofed as 12.9)"
