# `ibv_roundtrip` Notes (macOS Thunderbolt RDMA)

A latency measurement tool for RDMA over Thunderbolt on macOS. Performs ping-pong round-trip tests using UC (Unreliable Connection) transport.

**Current Version:** v0.0.23

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
# Server (e.g., rdma_en2 @ 10.1.12.4)
./ibv_roundtrip --server --listen 18515 --device rdma_en2 --ib-port 1 --gid-index 1 --iterations 1000
```

### Run Client
```bash
# Client (e.g., rdma_en5 @ 10.1.12.3)
./ibv_roundtrip --client --connect 10.1.12.4:18515 --device rdma_en5 --ib-port 1 --gid-index 1 --iterations 1000
```

### Expected Output (Client)
```
Client build v0.0.23
Registered buffers: recv lkey=0 send lkey=1
RTR: remote_qp=2304 remote_lid=1 psn=16759439 use_gid=1 mtu=1024
RTR: Using GRH with remote GID ::ffff:10.1.12.4 hop_limit=1
QP transitioned to RTR successfully
QP transitioned to RTS successfully
Client ready, QP in RTS state, starting to send...
Synchronized with server, starting to send

=== Latency Statistics (excluding first 100 warmup iterations) ===
Total iterations: 1000
Measured samples: 900
Min:              6.00 us
Avg:              9.18 us
Max:              14.00 us
```

The tool performs a warmup of 100 iterations, then measures round-trip latency for the remaining iterations. Statistics show min/avg/max in microseconds.

## Performance Characteristics

**Typical Latency (Thunderbolt RDMA, 1000-byte messages):**
- **Round-trip time:** ~9 μs average (6-14 μs range)
- **One-way latency:** ~4.5 μs (RTT/2)

**Comparison with Other Transports:**
- Dedicated RDMA NICs (RoCE/InfiniBand): 1-5 μs RTT
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
- **Message size:** 1000 bytes
- **Buffer size:** 4096 bytes (page-aligned)
- **Hop limit:** 1 (single L2 network)
- **Service Level:** 0

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
- **v0.0.23:** Removed all per-iteration debug output for clean statistics
- **v0.0.20-v0.0.22:** Added latency statistics with warmup period
- **v0.0.19:** Added synchronization barrier to fix packet delivery
- **v0.0.18:** Matched jaccl memory registration flags
- **v0.0.13:** Fixed "Bad address" error by reordering buffer allocation
