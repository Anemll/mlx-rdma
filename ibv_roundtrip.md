# `ibv_roundtrip` Notes (macOS Thunderbolt RDMA)

A latency and bandwidth measurement tool for RDMA over Thunderbolt on macOS. Performs ping-pong round-trip tests and flood bandwidth tests using UC (Unreliable Connection) transport.

**Current Version:** v0.0.61

## Constraints & Supported Features
- **Transport:** Apple's Thunderbolt RDMA adapters only expose UD/UC transports. The tool uses **UC queue pairs** just like `mlx/distributed/jaccl/jaccl.cpp`. RC is not available and attempts to create RC QPs return `EOPNOTSUPP`.
- **GID usage:** LIDs are always `0x0001`, so global routing is required. The tool auto-detects the correct GID index by searching for IPv4-mapped addresses (`::ffff:x.x.x.x`). Use `--gid-index N` to override if needed.
- **Queue pair limits:** `ibv_query_device` reports tiny capacities (≈11 QPs / CQEs). Creation will fail if other jobs hold QPs; stop MLX or reboot to free them.
- **Page-aligned buffers:** The hardware expects page-aligned, >=4 KB buffers. Use the provided helper that mirrors Jaccl's `page_aligned_alloc()` and register both send/recv buffers with full RW access.
- **Memory registration flags:** Must use `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE` to match jaccl configuration, even for SEND/RECV operations.
- **Synchronization barrier:** Both client and server synchronize via TCP after QP transitions to ensure both are ready before RDMA traffic begins. This prevents packet loss during initialization.
- **Server auto-restart:** After each test, the server fully restarts the RDMA device (closes ctx/pd, re-gets device list) to work around Apple driver resource leaks. A 2-second idle timeout handles packet loss gracefully.

## Common Pitfalls (Fixed in v0.0.23)
- **"Operation not supported" during `ibv_create_qp`:** Happens if you request RC/UD or exceed device caps for max WR/SQE. Keep WR counts ≤ `ibv_query_device().max_qp_wr` and use UC.
- **RTR failures (`status=60`):** Indicates GRH wasn't configured. Ensure both peers exchange nonzero GIDs (`--gid-index 1`) so `ibv_modify_qp` sets `ah_attr.is_global = 1`.
- **`ibv_post_recv: Bad address (-14)`:** Fixed by:
  1. Allocating and registering memory BEFORE QP state transitions (not after)
  2. Using full access flags: `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE`
  3. Posting receive buffers AFTER QP reaches RTS state
- **Packets not delivered (send completes but recv times out):** Fixed by adding synchronization barrier after both sides reach RTS state. Without this, client can send before server is ready.
- **Control connection errors:** The TCP side channel must target the server's IP/port, not the local host. Without it, QP info can't be exchanged.

## Build & Usage

### Build (with optimization)
```bash
c++ -std=c++17 -O2 -Wall -Wextra -o ibv_roundtrip ibv_roundtrip.cpp -lrdma -pthread
```

### Run Server
```bash
# Server (e.g., rdma_en2 @ 10.1.12.4) - no iterations/message-size needed, uses client's values
./ibv_roundtrip --server --listen 18515 --device rdma_en2 --ib-port 1 --gid-index 1
```

### Run Client (Ping-Pong Mode)
```bash
# Client (e.g., rdma_en5 @ 10.1.12.3)
./ibv_roundtrip --client --connect 10.1.12.4:18515 --device rdma_en5 --ib-port 1 --gid-index 1 --iterations 10000
```

### Run Client (Flood Mode - Bandwidth Test)
```bash
# Flood mode with 1 packet in flight per QP
./ibv_roundtrip --client --connect 10.1.12.4:18515 --device rdma_en5 --iterations 100000 --message-size 4096 --flood 1

# Flood mode with 4 packets in flight per QP (higher bandwidth, more loss)
./ibv_roundtrip --client --connect 10.1.12.4:18515 --device rdma_en5 --iterations 100000 --message-size 4096 --flood 4
```

### Run Client (Multi-QP with Threading)
```bash
# Multi-QP threaded mode - one thread per QP for maximum bandwidth
./ibv_roundtrip --client --connect 10.1.12.4:18515 --device rdma_en5 --iterations 1000000 --message-size 4096 --flood 1 --qp 6 --threaded --save
```

### Command Line Options
| Option | Description |
|--------|-------------|
| `--server` | Run in server mode |
| `--client` | Run in client mode |
| `--listen <port>` | Server listen port |
| `--connect <host:port>` | Server address to connect to |
| `--device <name>` | RDMA device name (e.g., rdma_en2, rdma_en5) |
| `--ib-port <n>` | IB port number (default: 1) |
| `--gid-index <n>` | GID index (use 1 for Thunderbolt RoCE) |
| `--iterations <n>` | Number of messages to send (client only) |
| `--message-size <bytes>` | Message size in bytes (client only, default: 1000, max: 65536) |
| `--flood [n]` | Flood mode with n packets in flight per QP (client only, default: max WR-2) |
| `--qp <n>` | Number of queue pairs to use (1-8, default: 1) |
| `--shared-cq` | Use single shared CQ for all QPs (reduces polling overhead) |
| `--threaded` | Use separate thread per QP (requires --flood and --qp > 1) |
| `--save` | Save results to results.json (append mode) |

**Note:** Server automatically uses client's `--iterations`, `--message-size`, and `--qp` values (synchronized via TCP control channel).

## Performance Results (v0.0.61)

### Benchmark: flood=1, 4096-byte messages, 1M iterations

| QPs | Mode | GB/s | Gbps | Avg Latency (µs) | Msg Rate | Loss |
|-----|------|------|------|------------------|----------|------|
| 1 | single | 2.77 | 22.2 | 12.36 | 338K | 0.0006% |
| 2 | threaded | 4.75 | 38.0 | 32.30 | 579K | 0% |
| 3 | threaded | 5.13 | 41.0 | 44.17 | 627K | 0% |
| 4 | threaded | 5.77 | 46.1 | 52.29 | 704K | 0% |
| 5 | threaded | 5.95 | 47.6 | 63.06 | 726K | 0% |
| 6 | threaded | 6.05 | 48.4 | 74.11 | 738K | 0% |
| 7 | threaded | 6.10 | 48.8 | 85.24 | 745K | 0% |

### Best Configuration
- **Maximum Bandwidth:** 6.14 GB/s (49.1 Gbps) @ 7 QPs threaded
- **Lowest Latency:** 12.36 µs @ 1 QP single-threaded
- **Best Balance:** 6 QPs threaded (~6 GB/s @ 74 µs latency)

### Recommended Usage
```bash
# For maximum throughput (6 GB/s)
./ibv_roundtrip --client --connect <host>:18515 --iterations 1000000 --message-size 4096 --flood 1 --qp 6 --threaded --save

# For lowest latency (12 µs)
./ibv_roundtrip --client --connect <host>:18515 --iterations 100000 --message-size 4096 --flood 1 --qp 1 --save
```

## Performance Characteristics

**Typical Latency (Thunderbolt RDMA, 4096-byte messages):**
- **Single QP:** ~12 μs average
- **Multi-QP threaded:** 30-85 μs average (scales with QP count)

**Typical Bandwidth (Thunderbolt RDMA, 4096-byte messages):**
- **Single QP:** ~2.8 GB/s (22 Gbps)
- **6 QPs threaded:** ~6.0 GB/s (48 Gbps)
- **7 QPs threaded:** ~6.1 GB/s (49 Gbps)

**Comparison with Other Transports:**
- Dedicated RDMA NICs (RoCE/InfiniBand): 1-5 μs RTT, 10+ GB/s
- TCP/IP localhost: 20-50 μs RTT
- 1Gb Ethernet: 100-500 μs RTT

## Implementation Details

### QP Setup Order (Critical!)
1. Allocate and register memory buffers
2. Exchange QP info via TCP control channel
3. Transition QP: RESET → INIT → RTR → RTS
4. Post receive buffers
5. **Synchronize via TCP barrier** (both sides ready)
6. Begin RDMA send/receive operations

### Configuration
- **Transport:** UC (Unreliable Connection)
- **MTU:** 4096 bytes (increased from 1024 in v0.0.60)
- **Message size:** 1000 bytes (default), configurable up to 64KB
- **Buffer size:** page-aligned (minimum 4096 bytes)
- **CQ size:** 256 entries (scaled for shared CQ mode)
- **QP capacity:** 64 send/recv work requests
- **Server receive window:** 8 pre-posted buffers (sliding window)
- **Hop limit:** 1 (single L2 network)
- **Service Level:** 0

### Test Modes
- **Ping-pong mode (default):** Send message, wait for echo, measure RTT. Best for latency measurement.
- **Flood mode (`--flood [n]`):** Send n messages in flight without waiting for echoes. Waits for all echoes at the end. Best for bandwidth measurement.
- **Threaded mode (`--threaded`):** One thread per QP for true parallel processing. Best for maximum throughput.
- **Shared CQ mode (`--shared-cq`):** Single CQ for all QPs to reduce polling overhead. Incompatible with threaded mode.

### JSON Results Format (`--save`)
Results are appended to `results.json`:
```json
{
  "timestamp": "2025-11-26 09:20:27",
  "version": "v0.0.61",
  "num_qps": 4,
  "iterations": 1000000,
  "message_size": 4096,
  "flood_outstanding": 1,
  "threaded": true,
  "shared_cq": false,
  "latency_us": { "min": 15.00, "avg": 52.29, "max": 488.00 },
  "bandwidth_gbps": 5.7673,
  "message_rate": 704016,
  "packets": { "sent": 1000000, "received": 1000000, "lost": 0, "loss_rate_pct": 0.0 },
  "duration_s": 1.420
}
```

## Debug Tips
- The tool logs `ibv_modify_qp` transitions with status codes and `strerror(errno)`. Use these to identify whether a failure is a capability (`EOPNOTSUPP`), address (`EFAULT`), or resource (`ENOMEM`) issue.
- If queue pair creation keeps failing, inspect `ibv_devinfo -d <device>` to see remaining QP capacity and ensure no other process is holding QPs.
- Use `ibv_devinfo` to verify GID index 1 contains a valid IPv4-mapped IPv6 address (e.g., `::ffff:10.1.12.4`).

## Known Limitations
- UC transport provides no retransmission; packet loss will cause test failure
- Maximum ~11 QPs per device due to Thunderbolt RDMA hardware constraints
- Requires two separate machines connected via Thunderbolt network
- macOS only (uses Apple's rdma library)
- Message sizes > MTU (4096) may cause fragmentation issues with UC transport

## Version History
- **v0.0.61:** Added `--save` option to save results to JSON file (append mode)
- **v0.0.60:** Increased MTU from 1024 to 4096 bytes to avoid fragmentation issues
- **v0.0.59:** Added `--threaded` option for separate thread per QP (true parallel processing)
- **v0.0.58:** Added `--shared-cq` option for single shared CQ across all QPs
- **v0.0.57:** Added stderr suppression during cleanup to hide Apple driver IOConnectUnmapMemory errors
- **v0.0.56:** Fixed cleanup order - destroy QP before MR deregistration (Apple driver requirement)
- **v0.0.55:** Fixed IOConnectUnmapMemory errors by reordering cleanup operations
- **v0.0.54:** Full device list cleanup on restart, increased restart delay to 500ms for better hardware reset
- **v0.0.53:** Added 2-second server idle timeout to prevent hanging on packet loss, enables auto-restart
- **v0.0.52:** Server auto-restarts RDMA device after each test (closes/reopens ctx, pd) to handle Apple driver quirks
- **v0.0.48-v0.0.51:** Added QP state transitions (RTS→ERROR→RESET) before destruction for clean cleanup
- **v0.0.47:** Fixed drain loop to poll all CQs together preventing hangs in multi-QP mode
- **v0.0.46:** Increased max QPs from 4 to 8
- **v0.0.45:** Fixed flood mode: `--flood N` now means N packets per QP (total = N × num_qps)
- **v0.0.44:** Fixed multi-QP flood mode by polling all CQs for completions
- **v0.0.43:** Added GID auto-detection for IPv4-mapped addresses across different devices
- **v0.0.41:** Added multi-QP support (`--qp N`) to increase bandwidth by using multiple parallel queue pairs
- **v0.0.37:** Added `--flood [n]` to specify max outstanding packets
- **v0.0.36:** Flood mode now tracks RTT for received echoes
- **v0.0.35:** Increased CQ (256) and QP capacity (64), fixed flood mode send completion issues
- **v0.0.33-v0.0.34:** Added flood mode debug output, timeout detection
- **v0.0.31-v0.0.32:** Added flood mode (`--flood`) for bandwidth testing
- **v0.0.29-v0.0.30:** Server sliding window (8 recv buffers), test parameter exchange, server uses client's iterations/message-size
- **v0.0.26:** Configurable message size (`--message-size`), packet loss detection, bandwidth statistics
- **v0.0.23:** Removed all per-iteration debug output for clean statistics
- **v0.0.20-v0.0.22:** Added latency statistics with warmup period
- **v0.0.19:** Added synchronization barrier to fix packet delivery
- **v0.0.18:** Matched jaccl memory registration flags
- **v0.0.13:** Fixed "Bad address" error by reordering buffer allocation
