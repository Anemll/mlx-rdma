# JACCL Distributed MLP Test - FIXED

## Status: Build Successful ✅

MLX with JACCL support has been successfully built and installed!

## Important: JACCL Requires 2+ Nodes

**JACCL is a distributed communication library** - it requires multiple processes/nodes running simultaneously to initialize. A single process cannot complete initialization because it waits at a barrier for other ranks to connect.

### Why Single Node Fails

When you run:
```bash
./run_jaccl.sh 0 10.1.12.4:1234  # Rank 0 alone
```

The `ConnectionManager` tries to:
1. Connect to coordinator (10.1.12.4:1234)
2. Exchange connection info with other ranks
3. Call `cm.barrier()` - **BLOCKS WAITING FOR RANK 1**

Since rank 1 never connects, initialization times out → "Cannot initialize jaccl distributed backend"

## How to Run (2-Node Setup Required)

### Prerequisites
- Both `m4p.local` and `m3u.local` must be running
- Both must have MLX with JACCL installed
- RDMA devices must be active (`rdma_en2` on m4p, `rdma_en5` on m3u)

### Step 1: Deploy Code to Both Hosts

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma/tests_jaccl
./deploy.sh
```

### Step 2: Start Both Ranks Simultaneously

**Terminal 1 (m4p.local - Rank 0):**
```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma/tests_jaccl
./run_jaccl.sh 0 10.1.12.4:1234
```

**Terminal 2 (m3u.local - Rank 1):**
```bash
ssh m3u.local
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma/tests_jaccl
./run_jaccl.sh 1 10.1.12.4:1234
```

Both processes will:
1. Initialize JACCL backend
2. Create distributed MLP model
3. Perform tensor-parallel forward pass
4. Verify output matches reference

### Expected Output

**Rank 0:**
```
Rank 0/2 initialized
Rank 0: Output shape (4, 16)
Rank 0: Output sample [...]
SUCCESS: Distributed output matches local output.
```

**Rank 1:**
```
Rank 1/2 initialized
Rank 1: Output shape (4, 16)
Rank 1: Output sample [...]
```

## Configuration Files

### [jaccl_config.json](jaccl_config.json) - Updated Format

```json
[
    [null, "rdma_en2"],    // Rank 0: null=self, rdma_en2 to connect to rank 1
    ["rdma_en5", null]     // Rank 1: rdma_en5 to connect to rank 0, null=self
]
```

Format explanation:
- Array index = rank number
- Inner array = devices for connecting to each peer
- `null` = self-connection (no device needed)
- Device name = RDMA interface to use for that peer

## Troubleshooting

### Error: "Cannot initialize jaccl distributed backend"

**Cause**: Only one rank is running

**Solution**: Start both ranks within a few seconds of each other

### Error: Connection timeout

**Causes**:
1. Firewall blocking port 1234
2. RDMA device down
3. Wrong IP address in coordinator

**Check**:
```bash
# Verify RDMA device is ACTIVE
./discover_ibv_devices

# Test network connectivity
nc -zv 10.1.12.4 1234
```

### Error: "Could not open device rdma_en5"

**Cause**: Wrong device name in config or device down

**Solution**: Run `discover_ibv_devices` on each host and update `jaccl_config.json`

## Single-Node Testing Alternative

If you only have one machine, JACCL cannot be tested. Consider using the `ring` backend instead:

```python
# In test_tp_mlp.py, change:
world = dist.init(strict=True, backend="ring")
```

The `ring` backend supports single-node multi-process testing via shared memory.

## Verification Commands

```bash
# Check JACCL is compiled in
nm ../build/libmlx.a | grep -i jaccl | head -5

# Verify is_available returns True
python -c "import mlx.core as mx; print(f'JACCL available: {mx.distributed.is_available()}')"

# Check RDMA devices
../discover_ibv_devices
```

## Build Information

- **MLX Version**: 0.30.1.dev20251129+a176c370
- **JACCL Support**: ✅ Enabled
- **macOS Version**: 26.2 (required >= 26.2)
- **Python Version**: 3.13.5 (required >= 3.10)
- **RDMA Devices**: rdma_en2, rdma_en3, rdma_en4
