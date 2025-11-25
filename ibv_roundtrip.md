# `ibv_roundtrip` Notes (macOS Thunderbolt RDMA)

A latency and bandwidth measurement tool for RDMA over Thunderbolt on macOS. Performs ping-pong round-trip tests and flood bandwidth tests using UC (Unreliable Connection) transport.

**Current Version:** v0.0.37

## Constraints & Supported Features
- **Transport:** Apple's Thunderbolt RDMA adapters only expose UD/UC transports. The tool uses **UC queue pairs** just like `mlx/distributed/jaccl/jaccl.cpp`. RC is not available and attempts to create RC QPs return `EOPNOTSUPP`.
- **GID usage:** LIDs are always `0x0001`, so global routing is required. Always run with `--gid-index 1` (Thunderbolt RoCE entry). Index 0 is `::` and causes RTR failures.
- **Queue pair limits:** `ibv_query_device` reports tiny capacities (≈11 QPs / CQEs). Creation will fail if other jobs hold QPs; stop MLX or reboot to free them.
- **Page-aligned buffers:** The hardware expects page-aligned, >=4 KB buffers. Use the provided helper that mirrors Jaccl's `page_aligned_alloc()` and register both send/recv buffers with full RW access.
- **Memory registration flags:** Must use `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE` to match jaccl configuration, even for SEND/RECV operations.
- **Synchronization barrier:** Both client and server synchronize via TCP after QP transitions to ensure both are ready before RDMA traffic begins. This prevents packet loss during initialization.

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
c++ -std=c++17 -O2 -Wall -Wextra -o ibv_roundtrip ibv_roundtrip.cpp -lrdma
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
# Flood mode with default 2 packets in flight
./ibv_roundtrip --client --connect 10.1.12.4:18515 --device rdma_en5 --ib-port 1 --gid-index 1 --iterations 100000 --message-size 4000 --flood

# Flood mode with 4 packets in flight (higher bandwidth)
./ibv_roundtrip --client --connect 10.1.12.4:18515 --device rdma_en5 --ib-port 1 --gid-index 1 --iterations 100000 --message-size 4000 --flood 4
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
| `--flood [n]` | Flood mode with n packets in flight (client only, default: 2) |

**Note:** Server automatically uses client's `--iterations` and `--message-size` values (synchronized via TCP control channel).

### Expected Output (Ping-Pong Mode)
```
Client build v0.0.37
Created CQ with 256 entries
Sent test parameters to server: iterations=10000 message_size=1000
...
=== Latency Statistics (excluding first 100 warmup iterations) ===
Total iterations: 10000
Measured samples: 9900
Min:              6.00 us
Avg:              14.19 us
Max:              115.00 us

=== Packet Loss Statistics ===
Total sent:       10000
Total received:   10000
Lost packets:     0
Out-of-order:     0
Duplicates:       0
Loss rate:        0.0000 %

=== Bandwidth Statistics ===
Test duration:    0.153 s
Message size:     1000 bytes
Total data:       0.02 GB (send+recv)
Bandwidth:        0.1306 GB/s
Message rate:     65306 msg/s
```

### Expected Output (Flood Mode)
```
Client build v0.0.37
FLOOD MODE: sending without waiting for echoes (max_outstanding=2)
  Sent 10000/100000 messages
  ...
Flood complete: sent 100000, received 99993 echoes

=== Latency Statistics (excluding first 100 warmup iterations) ===
Total iterations: 100000
Measured samples: 99893
Min:              9.00 us
Avg:              20.39 us
Max:              96.00 us

=== Bandwidth Statistics ===
Test duration:    0.523 s
Message size:     4000 bytes
Total data:       0.80 GB (send+recv)
Bandwidth:        1.5311 GB/s
Message rate:     191385 msg/s
```

The tool performs a warmup of 100 iterations, then measures round-trip latency for the remaining iterations. Statistics show min/avg/max in microseconds.

## Performance Characteristics

**Typical Latency (Thunderbolt RDMA, 1000-byte messages):**
- **Ping-pong RTT:** ~14 μs average (6-115 μs range)
- **Flood mode RTT:** ~20 μs average (9-96 μs range)
- **One-way latency:** ~7 μs (RTT/2)

**Typical Bandwidth (Thunderbolt RDMA, 4000-byte messages):**
- **Ping-pong mode:** ~0.55 GB/s, ~69K msg/s
- **Flood mode (2 outstanding):** ~1.5 GB/s, ~190K msg/s
- **Flood mode (4 outstanding):** ~1.7 GB/s, ~215K msg/s

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
- **MTU:** 1024 bytes
- **Message size:** 1000 bytes (default), configurable up to 64KB
- **Buffer size:** page-aligned (minimum 4096 bytes)
- **CQ size:** 256 entries
- **QP capacity:** 64 send/recv work requests
- **Server receive window:** 8 pre-posted buffers (sliding window)
- **Hop limit:** 1 (single L2 network)
- **Service Level:** 0

### Test Modes
- **Ping-pong mode (default):** Send message, wait for echo, measure RTT. Best for latency measurement.
- **Flood mode (`--flood [n]`):** Send n messages in flight without waiting for echoes. Waits for all echoes at the end. Best for bandwidth measurement.

## Debug Tips
- The tool logs `ibv_modify_qp` transitions with status codes and `strerror(errno)`. Use these to identify whether a failure is a capability (`EOPNOTSUPP`), address (`EFAULT`), or resource (`ENOMEM`) issue.
- If queue pair creation keeps failing, inspect `ibv_devinfo -d <device>` to see remaining QP capacity and ensure no other process is holding QPs.
- Use `ibv_devinfo` to verify GID index 1 contains a valid IPv4-mapped IPv6 address (e.g., `::ffff:10.1.12.4`).

## Known Limitations
- UC transport provides no retransmission; packet loss will cause test failure
- Maximum ~11 QPs per device due to Thunderbolt RDMA hardware constraints
- Requires two separate machines connected via Thunderbolt network
- macOS only (uses Apple's rdma library)

## Version History
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
