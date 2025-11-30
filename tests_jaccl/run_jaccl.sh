#!/bin/bash

# Example usage:
# ./run_jaccl.sh <rank> <coordinator_ip:port>

RANK=$1
COORDINATOR=$2

if [ -z "$RANK" ] || [ -z "$COORDINATOR" ]; then
    echo "Usage: $0 <rank> <coordinator_ip:port>"
    exit 1
fi

export MLX_RANK=$RANK
export MLX_IBV_COORDINATOR=$COORDINATOR
export MLX_IBV_DEVICES=$(pwd)/jaccl_config.json
export MLX_IBV_VERBOSE=1

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

python3 test_tp_mlp.py
