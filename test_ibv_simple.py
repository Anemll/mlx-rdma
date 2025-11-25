#!/usr/bin/env python3
"""
Simple test script for InfiniBand (IBV) backend.

Usage:
    export MLX_RANK=0
    export MLX_IBV_DEVICES=/path/to/devices.json
    export MLX_IBV_COORDINATOR=127.0.0.1:12345
    python test_ibv_simple.py
"""

import mlx.core as mx
import os
import sys


def main():
    # Check if backend is available
    if not mx.distributed.is_available("jaccl"):
        print("ERROR: JACCL backend is not available")
        print("\nRequirements:")
        print("  - macOS 26.2+")
        print("  - InfiniBand devices")
        print("  - Environment variables:")
        print("    * MLX_RANK")
        print("    * MLX_IBV_DEVICES")
        print("    * MLX_IBV_COORDINATOR")
        sys.exit(1)

    # Check environment variables
    rank = os.getenv("MLX_RANK")
    dev_file = os.getenv("MLX_IBV_DEVICES")
    coordinator = os.getenv("MLX_IBV_COORDINATOR")

    if not rank or not dev_file or not coordinator:
        print("ERROR: Missing required environment variables")
        print(f"  MLX_RANK: {rank}")
        print(f"  MLX_IBV_DEVICES: {dev_file}")
        print(f"  MLX_IBV_COORDINATOR: {coordinator}")
        sys.exit(1)

    print(f"[Rank {rank}] Initializing JACCL backend...")
    try:
        world = mx.distributed.init(backend="jaccl", strict=True)
        print(f"[Rank {rank}] Initialized: rank {world.rank()} / {world.size()}")
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize: {e}")
        sys.exit(1)

    # Test 1: Simple all_sum
    print(f"\n[Rank {world.rank()}] Test 1: all_sum")
    x = mx.ones(10) * (world.rank() + 1)
    print(f"[Rank {world.rank()}] Input: {x[0].item()}")
    
    y = mx.distributed.all_sum(x)
    mx.eval(y)
    expected_sum = sum(range(1, world.size() + 1))
    print(f"[Rank {world.rank()}] Result: {y[0].item()}, Expected: {expected_sum}")
    
    if abs(y[0].item() - expected_sum) < 1e-6:
        print(f"[Rank {world.rank()}] ✓ all_sum test passed")
    else:
        print(f"[Rank {world.rank()}] ✗ all_sum test failed")
        sys.exit(1)

    # Test 2: all_gather
    print(f"\n[Rank {world.rank()}] Test 2: all_gather")
    x = mx.ones((5,)) * (world.rank() + 1)
    y = mx.distributed.all_gather(x)
    mx.eval(y)
    print(f"[Rank {world.rank()}] Result shape: {y.shape}, Expected: ({world.size() * 5},)")
    
    if y.shape[0] == world.size() * 5:
        print(f"[Rank {world.rank()}] ✓ all_gather test passed")
    else:
        print(f"[Rank {world.rank()}] ✗ all_gather test failed")
        sys.exit(1)

    # Test 3: all_max
    print(f"\n[Rank {world.rank()}] Test 3: all_max")
    x = mx.ones(10) * (world.rank() + 1)
    y = mx.distributed.all_max(x)
    mx.eval(y)
    expected_max = world.size()
    print(f"[Rank {world.rank()}] Result: {y[0].item()}, Expected: {expected_max}")
    
    if abs(y[0].item() - expected_max) < 1e-6:
        print(f"[Rank {world.rank()}] ✓ all_max test passed")
    else:
        print(f"[Rank {world.rank()}] ✗ all_max test failed")
        sys.exit(1)

    # Test 4: send/recv (if size >= 2)
    if world.size() >= 2:
        print(f"\n[Rank {world.rank()}] Test 4: send/recv")
        right = (world.rank() + 1) % world.size()
        left = (world.rank() + world.size() - 1) % world.size()
        
        x = mx.ones(10) * (world.rank() + 1)
        
        if world.rank() % 2 == 0:
            y = mx.distributed.send(x, right)
            z = mx.distributed.recv_like(y, left)
            mx.eval(y, z)
        else:
            z = mx.distributed.recv_like(x, left)
            y = mx.distributed.send(x, right)
            mx.eval(z, y)
        
        print(f"[Rank {world.rank()}] Sent: {x[0].item()}, Received: {z[0].item()}")
        if abs(z[0].item() - (left + 1)) < 1e-6:
            print(f"[Rank {world.rank()}] ✓ send/recv test passed")
        else:
            print(f"[Rank {world.rank()}] ✗ send/recv test failed")
            sys.exit(1)

    print(f"\n[Rank {world.rank()}] All tests passed! ✓")


if __name__ == "__main__":
    main()

