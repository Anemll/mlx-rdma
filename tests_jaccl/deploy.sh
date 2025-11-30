#!/bin/bash

# Syncs the current directory to the test hosts
# Usage: ./deploy.sh

REMOTE_PATH="/Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma"

echo "Deploying to m4p.local..."
rsync -avz --exclude '.git' --exclude 'build' ../ m4p.local:$REMOTE_PATH/

echo "Deploying to m3u.local..."
# Check if m3u is up first to avoid long timeout if it's still down
if ping -c 1 -W 1 m3u.local &> /dev/null; then
    rsync -avz --exclude '.git' --exclude 'build' ../ m3u.local:$REMOTE_PATH/
else
    echo "m3u.local seems to be down. Skipping."
fi

echo "Done."
