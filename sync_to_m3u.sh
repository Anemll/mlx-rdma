#!/bin/bash
# Sync files needed to rebuild MLX with JACCL on m3u.local

set -e

REMOTE_HOST="m3u.local"
REMOTE_PATH="/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma"

echo "=== Syncing MLX-RDMA to m3u.local ==="
echo ""

# Check if m3u is reachable
if ! ping -c 1 -W 1 $REMOTE_HOST &> /dev/null; then
    echo "ERROR: $REMOTE_HOST is not reachable"
    exit 1
fi

echo "✓ $REMOTE_HOST is reachable"
echo ""

# Files/directories to sync
echo "Syncing source code and build scripts..."

rsync -avz --progress \
  --exclude '.git' \
  --exclude 'build/*' \
  --exclude '.venv' \
  --exclude '*.o' \
  --exclude '*.a' \
  --exclude '*.so' \
  --exclude '__pycache__' \
  --exclude 'results.json' \
  --exclude 'results_plot.png' \
  --exclude 'ibv_roundtrip' \
  --exclude 'discover_ibv_devices' \
  --exclude 'check_rdma_peer_interface*' \
  --exclude 'tcp_roundtrip' \
  --exclude 'test_jaccl_init*' \
  ./ $REMOTE_HOST:$REMOTE_PATH/

echo ""
echo "=== Sync Complete ==="
echo ""
echo "Files synced to: $REMOTE_HOST:$REMOTE_PATH"
echo ""
echo "Next steps on m3u.local:"
echo "  1. ssh $REMOTE_HOST"
echo "  2. cd $REMOTE_PATH"
echo "  3. ./setup_and_build_jaccl.sh"
echo ""
echo "Or run these commands:"
echo "  ssh $REMOTE_HOST 'cd $REMOTE_PATH && ./setup_and_build_jaccl.sh'"
echo ""
