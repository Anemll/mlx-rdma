# Deploy MLX with JACCL to m3u.local

## Summary

MLX with JACCL support has been successfully built on m4p.local. Now we need to deploy the same build to m3u.local to enable distributed testing across both machines.

## Prerequisites on m3u.local

1. Xcode-beta installed at `/Applications/Xcode-beta.app`
2. Command Line Tools installed
3. Python 3.10+ available
4. Thunderbolt RDMA interface available (rdma_en5)

## Deployment Steps

### Step 1: Sync Files from m4p to m3u

On **m4p.local**, run:

```bash
./sync_to_m3u.sh
```

This will sync all source files, build scripts, and tests to m3u.local.

### Step 2: Create Library Symlinks on m3u

On **m3u.local**, run:

```bash
ssh m3u.local
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma
./create_library_symlinks.sh
```

This creates symlinks for RDMA libraries from CommandLineTools SDK to Xcode-beta SDK:
- `/usr/lib/librdma.tbd`
- `/usr/lib/rdma/` directory

**Requires sudo** - will prompt for password.

### Step 3: Build MLX with JACCL on m3u

On **m3u.local**, run:

```bash
./setup_and_build_jaccl.sh
```

This will:
1. Verify Xcode-beta is selected
2. Check for header and library symlinks
3. Build C++ library with `cmake .. -DMLX_BUILD_JACCL=ON`
4. Install Python package with `CMAKE_ARGS="-DMLX_BUILD_JACCL=ON"`

Build takes ~5-10 minutes.

### Step 4: Verify JACCL on m3u

On **m3u.local**, verify the build:

```bash
source .venv/bin/activate
python -c "import mlx.core as mx; print('JACCL available:', mx.distributed.is_available())"
```

Expected output: `JACCL available: True`

## Testing Distributed Setup

### Configuration File

Both machines use `tests_jaccl/jaccl_config.json`:

```json
[
    [null, "rdma_en2"],    // Rank 0 (m4p): rdma_en2 to connect to rank 1
    ["rdma_en5", null]     // Rank 1 (m3u): rdma_en5 to connect to rank 0
]
```

### Running the Test

1. **On m3u.local (rank 1)** - Start first to wait for coordinator:
```bash
cd tests_jaccl
./run_jaccl.sh 1 10.1.12.4:1234
```

2. **On m4p.local (rank 0)** - Start coordinator:
```bash
cd tests_jaccl
./run_jaccl.sh 0 10.1.12.4:1234
```

Replace `10.1.12.4` with the actual IP address of the coordinator machine.

### Expected Output

Successful test shows:
- Both ranks initializing JACCL
- RDMA connections established
- Distributed MLP forward/backward passes completing
- No "Cannot initialize jaccl distributed backend" error

## Troubleshooting

### "Cannot initialize jaccl distributed backend"

Check:
1. Library symlinks exist: `ls -la /Applications/Xcode-beta.app/.../usr/lib/ | grep rdma`
2. Python package built with JACCL: `otool -L python/mlx/lib/libmlx.dylib | grep rdma`
3. CMAKE_ARGS was set during pip install

### "Permission denied" during symlink creation

Run `create_library_symlinks.sh` with sudo access. The script will prompt for password.

### Build failures

1. Verify Xcode-beta selected: `xcode-select -p`
2. Check symlinks exist for both headers AND libraries
3. Review build log output

## Files Synced to m3u

- Source code: `mlx/`, `python/`, `CMakeLists.txt`, etc.
- Build scripts: `setup_and_build_jaccl.sh`, `create_library_symlinks.sh`
- Tests: `tests_jaccl/` directory
- Config: `tests_jaccl/jaccl_config.json`
- Docs: `JACCL_BUILD_SUCCESS.md`, `DEPLOY_TO_M3U.md`

## One-Line Deployment

From **m4p.local**, sync and build on m3u in one command:

```bash
./sync_to_m3u.sh && ssh m3u.local 'cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma && ./create_library_symlinks.sh && ./setup_and_build_jaccl.sh'
```

Note: This requires ssh key auth or will prompt for password multiple times.

## System Information

### m4p.local (Rank 0)
- RDMA Interface: `rdma_en2`
- Role: Coordinator
- IP: (to be configured)

### m3u.local (Rank 1)
- RDMA Interface: `rdma_en5`
- Role: Worker
- IP: (to be configured)

## Next Steps After Successful Build

1. Identify IP addresses for coordinator
2. Update coordinator IP in test commands
3. Run distributed MLP test
4. Benchmark RDMA performance
5. Test with larger models/tensors

## References

- Build success: [JACCL_BUILD_SUCCESS.md](JACCL_BUILD_SUCCESS.md)
- Main build script: [setup_and_build_jaccl.sh](setup_and_build_jaccl.sh)
- Sync script: [sync_to_m3u.sh](sync_to_m3u.sh)
- Test runner: [tests_jaccl/run_jaccl.sh](tests_jaccl/run_jaccl.sh)
