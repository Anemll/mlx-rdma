#!/bin/bash
# Complete setup and build for MLX with JACCL support
# This script symlinks InfiniBand headers into Xcode-beta SDK and builds MLX

set -e

echo "=== MLX JACCL Setup and Build ==="
echo ""

# Paths - Headers
CLI_VERBS="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/infiniband/verbs.h"
XCODE_IB_DIR="/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk/usr/include/infiniband"
XCODE_VERBS="$XCODE_IB_DIR/verbs.h"

# Paths - Libraries
CLI_LIBRDMA="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/librdma.tbd"
CLI_RDMA_DIR="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/rdma"
XCODE_LIB_DIR="/Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk/usr/lib"
XCODE_LIBRDMA="$XCODE_LIB_DIR/librdma.tbd"
XCODE_RDMA_DIR="$XCODE_LIB_DIR/rdma"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Check if we're using Xcode-beta
CURRENT_XCODE=$(xcode-select -p)
if [[ "$CURRENT_XCODE" != "/Applications/Xcode-beta.app/Contents/Developer" ]]; then
    echo "ERROR: Please switch to Xcode-beta first:"
    echo "  sudo xcode-select -s /Applications/Xcode-beta.app/Contents/Developer"
    exit 1
fi

echo "✓ Using Xcode-beta"

# Check if header symlink already exists
if [ -f "$XCODE_VERBS" ]; then
    echo "✓ InfiniBand header already present in Xcode-beta SDK"
else
    echo "Setting up InfiniBand headers in Xcode-beta SDK..."
    echo "This requires sudo to create files in Xcode SDK directory"

    # Create directory if it doesn't exist
    sudo mkdir -p "$XCODE_IB_DIR"

    # Create symlink
    sudo ln -sf "$CLI_VERBS" "$XCODE_VERBS"

    echo "✓ Header symlink created: $XCODE_VERBS -> $CLI_VERBS"
fi

# Verify the header is accessible
if [ ! -f "$XCODE_VERBS" ]; then
    echo "ERROR: InfiniBand header not found at $XCODE_VERBS"
    exit 1
fi

# Check if library symlinks already exist
if [ -f "$XCODE_LIBRDMA" ] && [ -d "$XCODE_RDMA_DIR" ]; then
    echo "✓ RDMA libraries already present in Xcode-beta SDK"
else
    echo "Setting up RDMA libraries in Xcode-beta SDK..."
    echo "This requires sudo to create files in Xcode SDK directory"

    # Create librdma.tbd symlink
    if [ ! -f "$XCODE_LIBRDMA" ]; then
        sudo ln -sf "$CLI_LIBRDMA" "$XCODE_LIBRDMA"
        echo "✓ Library symlink created: $XCODE_LIBRDMA -> $CLI_LIBRDMA"
    fi

    # Create rdma directory symlink
    if [ ! -d "$XCODE_RDMA_DIR" ]; then
        sudo ln -sf "$CLI_RDMA_DIR" "$XCODE_RDMA_DIR"
        echo "✓ Library directory symlink created: $XCODE_RDMA_DIR -> $CLI_RDMA_DIR"
    fi
fi

# Verify the libraries are accessible
if [ ! -f "$XCODE_LIBRDMA" ]; then
    echo "ERROR: librdma.tbd not found at $XCODE_LIBRDMA"
    exit 1
fi
if [ ! -d "$XCODE_RDMA_DIR" ]; then
    echo "ERROR: rdma directory not found at $XCODE_RDMA_DIR"
    exit 1
fi

echo ""
echo "=== Building MLX with JACCL ==="
echo ""

# Clean build directory
echo "Cleaning build directory..."
cd "$BUILD_DIR"
rm -rf ./*

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. -DMLX_BUILD_JACCL=ON

# Build
echo ""
echo "Building MLX with JACCL..."
make -j8

# Install Python package
echo ""
echo "Installing Python package..."
cd "$SCRIPT_DIR"

# Use existing venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Creating new venv..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Uninstall old MLX
pip uninstall mlx -y 2>/dev/null || true

# Install from source with JACCL enabled
echo "Installing MLX from source with JACCL support..."
export CMAKE_ARGS="-DMLX_BUILD_JACCL=ON"
pip install -e .

echo ""
echo "=== Build Complete ==="
echo ""
echo "Testing mlx.distributed import..."
python -c "
import mlx.core as mx
print('✓ mlx.distributed available:', hasattr(mx, 'distributed'))
print('✓ Functions:', dir(mx.distributed))
print('✓ JACCL backend should now work!')
"

echo ""
echo "MLX with JACCL support is installed!"
echo "Run tests with: cd tests_jaccl && ./run_jaccl.sh 0 10.1.12.4:1234"
echo ""
