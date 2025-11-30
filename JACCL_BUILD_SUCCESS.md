# JACCL Build SUCCESS

## Problem Solved

MLX with JACCL support has been successfully built and configured!

## Root Cause

The issue was that **Xcode-beta SDK was missing RDMA library files** (`librdma.tbd` and the `rdma/` directory), which prevented proper linking of InfiniBand Verbs symbols.

### What Was Missing

1. **Headers**: `/usr/include/infiniband/verbs.h` (fixed earlier with symlink)
2. **Libraries** (newly discovered):
   - `/usr/lib/librdma.tbd` - RDMA library descriptor
   - `/usr/lib/rdma/` - Directory containing `libibverbs.dylib`, `libmlx5.dylib`, etc.

## The Fix

### Step 1: Create Library Symlinks

Created symlinks from CommandLineTools SDK (which has RDMA support) to Xcode-beta SDK:

```bash
sudo ln -sf \
  /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/librdma.tbd \
  /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk/usr/lib/librdma.tbd

sudo ln -sf \
  /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/rdma \
  /Applications/Xcode-beta.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.0.sdk/usr/lib/rdma
```

### Step 2: Fix CMakeLists.txt Linking

Changed linking from `PRIVATE` to `PUBLIC` in `mlx/distributed/jaccl/CMakeLists.txt`:

```cmake
target_link_libraries(mlx PUBLIC rdma)  # Was: PRIVATE rdma
```

This ensures that when building shared libraries (like `libmlx.dylib` for Python), the rdma dependency is propagated.

### Step 3: Set CMAKE_ARGS for Python Build

The Python build system needs the `MLX_BUILD_JACCL` flag passed via environment variable:

```bash
export CMAKE_ARGS="-DMLX_BUILD_JACCL=ON"
pip install -e .
```

## Verification

### Check libmlx.dylib Links

```bash
$ otool -L python/mlx/lib/libmlx.dylib | grep rdma
/usr/lib/librdma.dylib (compatibility version 1.0.0, current version 1.0.0)
```

### Check JACCL Availability

```python
import mlx.core as mx
print(mx.distributed.is_available())  # Returns: True
```

## Updated Build Scripts

### [create_library_symlinks.sh](create_library_symlinks.sh)

One-time script to create necessary symlinks (requires sudo).

### [setup_and_build_jaccl.sh](setup_and_build_jaccl.sh)

Complete build script that:
1. Verifies Xcode-beta is selected
2. Checks for header and library symlinks
3. Builds C++ library with CMake `-DMLX_BUILD_JACCL=ON`
4. Installs Python package with `CMAKE_ARGS="-DMLX_BUILD_JACCL=ON"`

## System Requirements

- **macOS Version**: 26.2+ (Sequoia 16.2+) - JACCL runtime requirement
- **Build Tools**:
  - Xcode-beta (for Metal compiler)
  - Command Line Tools (for RDMA headers/libraries)
- **Hardware**: Mac with Thunderbolt RDMA support

## Next Steps

1. **Sync to m3u.local**: Use `./sync_to_m3u.sh` to copy files to second machine
2. **Build on m3u**: Run `./create_library_symlinks.sh` then `./setup_and_build_jaccl.sh`
3. **Test distributed**: Run 2-node JACCL test with `tests_jaccl/run_jaccl.sh`

## Files Modified

- `mlx/distributed/jaccl/CMakeLists.txt` - Changed `PRIVATE` to `PUBLIC` linking
- `setup_and_build_jaccl.sh` - Added `CMAKE_ARGS` export for Python build
- `create_library_symlinks.sh` - New script for library symlinks

## References

- RDMA libraries location: `/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib/`
- InfiniBand Verbs header: `infiniband/verbs.h`
- JACCL source: `mlx/distributed/jaccl/jaccl.cpp`
- Python build system: `setup.py` (reads `CMAKE_ARGS` environment variable)
