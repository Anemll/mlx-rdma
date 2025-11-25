# Testing InfiniBand (IBV) Backend

This guide explains how to test the new InfiniBand Verbs (ibv) backend in the `ibv-backend` branch.

## Prerequisites

1. **macOS 26.2 or later** - The backend requires macOS 26.2+ (check with `sw_vers`)
2. **InfiniBand hardware** - You need InfiniBand-capable network interfaces
3. **libibverbs** - The RDMA library should be available (linked as `rdma` in CMake)

### Checking Prerequisites

```bash
# Check macOS version
sw_vers

# Check if libibverbs is available (should find libibverbs.dylib)
find /usr -name "*ibverbs*" 2>/dev/null
find /opt -name "*ibverbs*" 2>/dev/null

# Or check if rdma library is linkable
c++ -lrdma -o /dev/null 2>&1 | head -1
```

**Note:** InfiniBand support on macOS is limited and requires:
- macOS 26.2+ (beta/preview versions)
- Compatible InfiniBand hardware
- Proper drivers and libraries installed

## Thunderbolt RDMA Setup

For Thunderbolt-based RDMA devices, you need to enable RDMA support:

1. **Enable RDMA in Recovery OS/Terminal:**
   ```bash
   # Boot into Recovery OS or use terminal with appropriate privileges
   rdma_ctl enable
   ```
   
2. **Reboot the system** after enabling RDMA

3. **Verify RDMA is enabled** using InfiniBand tools:
   ```bash
   # List available InfiniBand devices
   ibv_devices
   
   # Get detailed device information
   ibv_devinfo
   ```

These commands will help you identify your Thunderbolt RDMA devices and their capabilities.

## Step 1: Discover InfiniBand Devices

First, you need to find the names of your InfiniBand devices on macOS.

### Method 1: Use the Provided Utility Script

A C++ utility is provided in `discover_ibv_devices.cpp`. Compile and run it:

```bash
# Compile the utility
c++ -o discover_ibv_devices discover_ibv_devices.cpp -lrdma

# Run it to discover devices
./discover_ibv_devices
```

This will:
- List all available InfiniBand devices
- Show device information (vendor ID, device ID, capabilities)
- Generate example JSON configuration files

### Method 2: Use Python Discovery Script

```bash
python discover_ibv_devices.py
```

This script checks if MLX is available and provides guidance on device discovery.

### Method 3: System Information

Check system network information:

```bash
system_profiler SPNetworkDataType | grep -i infiniband
```

### Method 4: Use InfiniBand Tools (if available)

If you have InfiniBand tools installed:

```bash
# List available InfiniBand devices
ibv_devices

# Get detailed device information
ibv_devinfo
```

### Method 5: Manual C++ Program

You can also create a simple C++ program to list devices (see `discover_ibv_devices.cpp` for a complete example):

```cpp
#include <infiniband/verbs.h>
#include <iostream>

int main() {
    int num_devices;
    ibv_device** devices = ibv_get_device_list(&num_devices);
    
    if (!devices) {
        std::cerr << "No InfiniBand devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << num_devices << " InfiniBand device(s):" << std::endl;
    for (int i = 0; i < num_devices; i++) {
        std::cout << "  " << ibv_get_device_name(devices[i]) << std::endl;
    }
    
    ibv_free_device_list(devices);
    return 0;
}
```

Compile with: `c++ -o list_devices list_devices.cpp -lrdma`

### Understanding Device Discovery Output

When you run the device discovery tools, you'll see output like:

```
- Vendor ID: 0x0
- Device ID: 0x0
- Max MR size: 16384000 bytes
- Max QP: 11
- Max CQ: 11
- Port 1 LID: 3
- Port 1 State: 1
- Port 1 MTU: 5
```

**MTU (Maximum Transmission Unit) Decode:**
- `1` = 256 bytes
- `2` = 512 bytes
- `3` = 1024 bytes
- `4` = 2048 bytes
- `5` = 4096 bytes

**Port State Values:**
- `1` = `IBV_PORT_DOWN` (port is down/not active)
- `2` = `IBV_PORT_INIT` (port is initializing)
- `3` = `IBV_PORT_ARMED` (port is armed)
- `4` = `IBV_PORT_ACTIVE` (port is active and ready)

**Other Fields:**
- **Vendor ID / Device ID**: Hardware identifiers (0x0 may indicate virtual/emulated device)
- **Max MR size**: Maximum Memory Region size (registered memory for RDMA)
- **Max QP**: Maximum number of Queue Pairs (communication endpoints)
- **Max CQ**: Maximum number of Completion Queues (completion notifications)
- **LID**: Local Identifier (routing identifier for the port)

## Step 2: Create Device Configuration File

Create a JSON file (e.g., `ibv_devices.json`) that maps each rank to its InfiniBand device(s). 

**Key understanding:** The device file defines which InfiniBand device each rank should use to connect to each *other* rank. The rank's own entry (at its own index) should be `null` or empty string.

The format is:
```json
{
  "0": [null, "device_for_rank1", "device_for_rank2", ...],
  "1": ["device_for_rank0", null, "device_for_rank2", ...],
  "2": ["device_for_rank0", "device_for_rank1", null, ...]
}
```

**Important notes:**
- The keys are rank numbers as strings ("0", "1", "2", ...)
- Each rank has an array with `size` elements (where `size` is the total number of ranks)
- At index `i`, you specify the device name to use when connecting to rank `i`
- At your own rank's index, use `null` or empty string `""` (you don't connect to yourself)
- The array length must match the total number of ranks

**Example for 2-node setup:**
```json
{
  "0": [null, "mlx5_0"],
  "1": ["mlx5_0", null]
}
```

This means:
- Rank 0: connects to rank 1 using device "mlx5_0" (at index 1), null at index 0 (itself)
- Rank 1: connects to rank 0 using device "mlx5_0" (at index 0), null at index 1 (itself)

**Note:** In a typical setup, all ranks might use the same device name (e.g., "mlx5_0") to connect to each other, but each rank's own entry is null.

## Step 3: Set Environment Variables

For each process, you need to set:

1. **MLX_RANK** - The rank of this process (0, 1, 2, ...)
2. **MLX_IBV_DEVICES** - Path to the JSON device configuration file
3. **MLX_IBV_COORDINATOR** - IP address and port for the TCP side channel (format: `IP:PORT`)

**Example:**
```bash
export MLX_RANK=0
export MLX_IBV_DEVICES=/path/to/ibv_devices.json
export MLX_IBV_COORDINATOR=127.0.0.1:12345
```

## Step 4: Simple Test Script

Create a simple Python test script (`test_ibv_simple.py`):

```python
import mlx.core as mx
import os

# Initialize the jaccl backend
world = mx.distributed.init(backend="jaccl")
print(f"Rank {world.rank()} / {world.size()}")

# Simple all_sum test
x = mx.ones(10) * (world.rank() + 1)
print(f"Rank {world.rank()}: x = {x[0].item()}")

y = mx.distributed.all_sum(x)
mx.eval(y)
print(f"Rank {world.rank()}: all_sum result = {y[0].item()}")
print(f"Rank {world.rank()}: Expected sum = {sum(range(1, world.size() + 1))}")
```

## Step 5: Run the Test

### Option A: Manual Launch (2 processes on same machine)

**Terminal 1:**
```bash
export MLX_RANK=0
export MLX_IBV_DEVICES=/path/to/ibv_devices.json
export MLX_IBV_COORDINATOR=127.0.0.1:12345
python test_ibv_simple.py
```

**Terminal 2:**
```bash
export MLX_RANK=1
export MLX_IBV_DEVICES=/path/to/ibv_devices.json
export MLX_IBV_COORDINATOR=127.0.0.1:12345
python test_ibv_simple.py
```

### Option B: Using the Test Suite

Run the distributed test suite:

```bash
# Set environment variables for all processes
export MLX_IBV_DEVICES=/path/to/ibv_devices.json
export MLX_IBV_COORDINATOR=127.0.0.1:12345

# Launch 2 processes (adjust as needed)
MLX_RANK=0 python python/tests/jaccl_test_distributed.py &
MLX_RANK=1 python python/tests/jaccl_test_distributed.py &
wait
```

## Step 6: Verify Backend Availability

You can check if the backend is available:

```python
import mlx.core as mx

# Check if jaccl backend is available
if mx.distributed.is_available("jaccl"):
    print("JACCL backend is available")
    world = mx.distributed.init(backend="jaccl")
    print(f"Initialized with {world.size()} processes")
else:
    print("JACCL backend is not available")
    print("Requirements:")
    print("  - macOS 26.2+")
    print("  - InfiniBand devices")
    print("  - Environment variables: MLX_RANK, MLX_IBV_DEVICES, MLX_IBV_COORDINATOR")
```

## Troubleshooting

1. **"Could not open device"**: Check that the device name in the JSON file matches the actual device name
2. **"Connection attempt" messages**: The TCP side channel is trying to connect - ensure the coordinator address is correct
3. **"Malformed device file"**: Check the JSON format - rank's own device should be null/empty
4. **"Maximum number of supported peers"**: Currently limited to 8 peers (MAX_PEERS = 8)

## Example Device File for 2-Node Setup

```json
{
  "0": [null, "mlx5_0"],
  "1": ["mlx5_0", null]
}
```

## Example Device File for 4-Node Setup

```json
{
  "0": [null, "mlx5_0", "mlx5_1", "mlx5_2"],
  "1": ["mlx5_0", null, "mlx5_1", "mlx5_2"],
  "2": ["mlx5_0", "mlx5_1", null, "mlx5_2"],
  "3": ["mlx5_0", "mlx5_1", "mlx5_2", null]
}
```

Where each rank connects to all other ranks' devices, but not to its own (null at its own index).

