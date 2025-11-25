#!/usr/bin/env python3
"""
Python script to discover InfiniBand devices using MLX (if available).
This is a fallback if the C++ utility cannot be compiled.
"""

import sys
import os

def check_with_mlx():
    """Try to discover devices using MLX if it's installed."""
    try:
        import mlx.core as mx
        
        # Check if jaccl backend is available
        if not mx.distributed.is_available("jaccl"):
            print("JACCL backend is not available")
            print("\nRequirements:")
            print("  - macOS 26.2+")
            print("  - InfiniBand hardware")
            print("  - libibverbs library")
            return False
        
        print("JACCL backend is available!")
        print("\nTo discover devices, you need to:")
        print("  1. Compile discover_ibv_devices.cpp")
        print("  2. Or check system information")
        print("  3. Or use ibv_devices command if available")
        return True
        
    except ImportError:
        print("MLX is not installed")
        return False

def main():
    print("InfiniBand Device Discovery for macOS")
    print("=" * 50)
    print()
    
    # Check macOS version
    import platform
    mac_version = platform.mac_ver()[0]
    print(f"macOS version: {mac_version}")
    
    major_version = int(mac_version.split('.')[0]) if mac_version else 0
    if major_version < 26:
        print("\n⚠️  Warning: macOS 26.2+ is required for InfiniBand support")
        print("   Current version may not support InfiniBand Verbs")
    
    print("\nTo discover InfiniBand devices:")
    print("  1. Compile the C++ utility:")
    print("     c++ -o discover_ibv_devices discover_ibv_devices.cpp -lrdma")
    print("     ./discover_ibv_devices")
    print()
    print("  2. Or check system information:")
    print("     system_profiler SPNetworkDataType | grep -i infiniband")
    print()
    print("  3. Or use ibv_devices (if installed):")
    print("     ibv_devices")
    print()
    
    # Try MLX check
    if check_with_mlx():
        print("\n✓ MLX with JACCL backend is available")
        print("  You can proceed with testing once devices are configured")
    
    print("\nFor more information, see TEST_IBV.md")

if __name__ == "__main__":
    main()

