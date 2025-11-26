#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// RAII helper to suppress stderr during cleanup (hides Apple driver IOConnectUnmapMemory errors)
class StderrSuppressor {
  int saved_fd_ = -1;
  bool active_ = false;
public:
  StderrSuppressor() {
    // Save original stderr
    saved_fd_ = dup(STDERR_FILENO);
    if (saved_fd_ >= 0) {
      // Redirect stderr to /dev/null
      int null_fd = open("/dev/null", O_WRONLY);
      if (null_fd >= 0) {
        dup2(null_fd, STDERR_FILENO);
        close(null_fd);
        active_ = true;
      }
    }
  }
  ~StderrSuppressor() {
    if (active_ && saved_fd_ >= 0) {
      // Restore original stderr
      dup2(saved_fd_, STDERR_FILENO);
      close(saved_fd_);
    }
  }
  StderrSuppressor(const StderrSuppressor&) = delete;
  StderrSuppressor& operator=(const StderrSuppressor&) = delete;
};

constexpr std::size_t kDefaultMessageSize = 1000;
constexpr std::size_t kMaxMessageSize = 65536;  // 64KB max
constexpr const char* kBuildVersion = "v0.0.57";
constexpr int kMaxQPs = 8;  // Maximum number of queue pairs (hardware supports ~11)
constexpr int kMaxGIDIndex = 8;  // Max GID indices to search

// Number of receive buffers to pre-post (sliding window)
constexpr int kRecvWindowSize = 8;

struct MessageHeader {
  uint64_t sequence;
  uint64_t send_time_ns;
};

constexpr std::size_t kHeaderSize = sizeof(MessageHeader);

struct WireQPInfo {
  uint16_t lid;
  uint32_t qp_num;
  uint32_t psn;
  uint8_t gid[16];
};

// Exchanged after QP info to synchronize test parameters
struct WireTestParams {
  uint32_t iterations;
  uint32_t message_size;
  uint32_t num_qps;      // Number of queue pairs to use (1-4)
};

struct Options {
  enum class Mode { Server, Client } mode;
  std::string device_name;
  uint8_t ib_port = 1;
  uint8_t gid_index = 1;
  uint16_t listen_port = 0;
  std::string connect_host;
  uint16_t connect_port = 0;
  int iterations = 16;
  std::size_t message_size = kDefaultMessageSize;
  int flood_outstanding = -1;  // -1 = disabled (ping-pong), 0 = unlimited, >0 = limited outstanding
  int num_qps = 1;             // Number of queue pairs (1-4)
};

// Per-QP resources for multi-QP support
struct QPContext {
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  uint32_t psn = 0;

  // Client: single send/recv buffer per QP
  void* send_buf = nullptr;
  void* recv_buf = nullptr;
  ibv_mr* send_mr = nullptr;
  ibv_mr* recv_mr = nullptr;

  // Server: sliding window of receive buffers per QP
  std::vector<void*> recv_bufs;
  std::vector<ibv_mr*> recv_mrs;

  // Per-QP state
  int outstanding_sends = 0;
  uint64_t sequence = 0;
  int handled = 0;
};

void* allocate_page_aligned(size_t num_bytes) {
  static size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  void* ptr = nullptr;
  if (posix_memalign(&ptr, page_size, num_bytes) != 0) {
    return nullptr;
  }
  return ptr;
}

uint64_t wall_time_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

int parse_port(const std::string& value) {
  int port = std::stoi(value);
  if (port <= 0 || port > 65535) {
    throw std::runtime_error("Port must be in range 1-65535");
  }
  return port;
}

bool send_all(int fd, const void* data, size_t len) {
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  while (len > 0) {
    ssize_t sent = ::send(fd, ptr, len, 0);
    if (sent <= 0) {
      return false;
    }
    ptr += sent;
    len -= static_cast<size_t>(sent);
  }
  return true;
}

bool recv_all(int fd, void* data, size_t len) {
  uint8_t* ptr = static_cast<uint8_t*>(data);
  while (len > 0) {
    ssize_t recvd = ::recv(fd, ptr, len, 0);
    if (recvd <= 0) {
      return false;
    }
    ptr += recvd;
    len -= static_cast<size_t>(recvd);
  }
  return true;
}

int create_listen_socket(uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    perror("socket");
    return -1;
  }

  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    perror("bind");
    ::close(fd);
    return -1;
  }

  if (::listen(fd, 1) < 0) {
    perror("listen");
    ::close(fd);
    return -1;
  }

  return fd;
}

int accept_control(int listen_fd) {
  sockaddr_in addr{};
  socklen_t len = sizeof(addr);
  int fd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&addr), &len);
  if (fd < 0) {
    perror("accept");
  }
  return fd;
}

int connect_control(const std::string& host, uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  addrinfo* res = nullptr;
  int rv = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);
  if (rv != 0) {
    std::cerr << "getaddrinfo: " << gai_strerror(rv) << std::endl;
    return -1;
  }

  int fd = -1;
  for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) {
      continue;
    }
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
      break;
    }
    ::close(fd);
    fd = -1;
  }

  ::freeaddrinfo(res);
  if (fd < 0) {
    std::cerr << "Could not connect to control host" << std::endl;
  }
  return fd;
}

bool exchange_qp_info(int fd, WireQPInfo& local, WireQPInfo& remote, bool is_server) {
  if (is_server) {
    if (!send_all(fd, &local, sizeof(local))) {
      return false;
    }
    if (!recv_all(fd, &remote, sizeof(remote))) {
      return false;
    }
  } else {
    if (!recv_all(fd, &remote, sizeof(remote))) {
      return false;
    }
    if (!send_all(fd, &local, sizeof(local))) {
      return false;
    }
  }
  return true;
}

// Exchange test parameters - server receives client's params and uses them
bool exchange_test_params(int fd, WireTestParams& local, WireTestParams& remote, bool is_server) {
  if (is_server) {
    // Server receives client params first, then sends acknowledgment
    if (!recv_all(fd, &remote, sizeof(remote))) {
      return false;
    }
    // Server echoes back to confirm (using client's params)
    if (!send_all(fd, &remote, sizeof(remote))) {
      return false;
    }
  } else {
    // Client sends its params
    if (!send_all(fd, &local, sizeof(local))) {
      return false;
    }
    // Client receives server's acknowledgment
    if (!recv_all(fd, &remote, sizeof(remote))) {
      return false;
    }
  }
  return true;
}

bool is_gid_nonzero(const ibv_gid& gid) {
  for (uint8_t b : gid.raw) {
    if (b != 0) {
      return true;
    }
  }
  return false;
}

// Check if GID is an IPv4-mapped address (::ffff:x.x.x.x)
bool is_ipv4_mapped_gid(const ibv_gid& gid) {
  // IPv4-mapped: first 10 bytes zero, next 2 bytes 0xFF, last 4 bytes IPv4
  for (int i = 0; i < 10; ++i) {
    if (gid.raw[i] != 0) return false;
  }
  return gid.raw[10] == 0xFF && gid.raw[11] == 0xFF;
}

// Auto-detect GID index: find first IPv4-mapped GID, or use provided index
// Returns -1 if no valid GID found
int find_valid_gid_index(ibv_context* ctx, uint8_t port, int requested_index, ibv_gid& out_gid) {
  // First try the requested index
  if (ibv_query_gid(ctx, port, requested_index, &out_gid) == 0) {
    if (is_ipv4_mapped_gid(out_gid)) {
      return requested_index;
    }
  }

  // Requested index wasn't valid IPv4-mapped, search for one
  std::cout << "GID[" << requested_index << "] is not IPv4-mapped, searching..." << std::endl;
  for (int gi = 0; gi < kMaxGIDIndex; ++gi) {
    ibv_gid test_gid{};
    if (ibv_query_gid(ctx, port, gi, &test_gid) == 0) {
      if (is_ipv4_mapped_gid(test_gid)) {
        out_gid = test_gid;
        std::cout << "Found IPv4-mapped GID at index " << gi << std::endl;
        return gi;
      }
    }
  }
  return -1;  // No valid GID found
}

bool modify_qp_to_init(ibv_qp* qp, uint8_t port) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = port;
  attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE;

  int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  if (int status = ibv_modify_qp(qp, &attr, mask); status != 0) {
    int err = status < 0 ? -status : status;
    std::cerr << "ibv_modify_qp INIT failed, status=" << status << " ("
              << std::strerror(err) << ")" << std::endl;
    return false;
  }
  return true;
}

bool modify_qp_to_rtr(ibv_qp* qp, const WireQPInfo& remote, uint8_t port, bool use_gid,
                      uint8_t gid_index) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024;  // Match jaccl.cpp
  attr.dest_qp_num = remote.qp_num;
  attr.rq_psn = remote.psn;
  attr.ah_attr.is_global = use_gid;
  attr.ah_attr.dlid = remote.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = port;

  std::cout << "RTR: remote_qp=" << remote.qp_num << " remote_lid=" << remote.lid
            << " psn=" << remote.psn << " use_gid=" << use_gid
            << " mtu=1024" << std::endl;

  if (use_gid) {
    ibv_gid remote_gid{};
    std::memcpy(remote_gid.raw, remote.gid, 16);
    attr.ah_attr.grh.dgid = remote_gid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 1;  // Match jaccl.cpp
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.traffic_class = 0;

    char gid_str[40];
    inet_ntop(AF_INET6, remote_gid.raw, gid_str, sizeof(gid_str));
    std::cout << "RTR: Using GRH with remote GID " << gid_str
              << " hop_limit=1" << std::endl;
  }

  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
  if (int status = ibv_modify_qp(qp, &attr, mask); status != 0) {
    int err = status < 0 ? -status : status;
    std::cerr << "ibv_modify_qp RTR failed, status=" << status << " ("
              << std::strerror(err) << ")" << std::endl;
    return false;
  }
  std::cout << "QP transitioned to RTR successfully" << std::endl;
  return true;
}

bool modify_qp_to_rts(ibv_qp* qp, uint32_t psn) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = psn;
  int mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  if (int status = ibv_modify_qp(qp, &attr, mask); status != 0) {
    int err = status < 0 ? -status : status;
    std::cerr << "ibv_modify_qp RTS failed, status=" << status << " ("
              << std::strerror(err) << ")" << std::endl;
    return false;
  }
  std::cout << "QP transitioned to RTS successfully" << std::endl;
  return true;
}

bool post_receive(ibv_qp* qp, void* buffer, ibv_mr* mr, uint64_t wr_id, std::size_t msg_size) {
  ibv_sge sg{};
  sg.addr = reinterpret_cast<uintptr_t>(buffer);
  sg.length = static_cast<uint32_t>(msg_size);
  sg.lkey = mr->lkey;

  ibv_recv_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sg;
  wr.num_sge = 1;

  ibv_recv_wr* bad = nullptr;
  int status = ibv_post_recv(qp, &wr, &bad);
  if (status != 0) {
    int err = status < 0 ? -status : status;
    std::cerr << "ibv_post_recv failed, status=" << status << " ("
              << std::strerror(err) << ")" << std::endl;
    return false;
  }
  return true;
}

bool post_send(ibv_qp* qp, void* buffer, ibv_mr* mr, uint64_t wr_id, std::size_t msg_size) {
  ibv_sge sg{};
  sg.addr = reinterpret_cast<uintptr_t>(buffer);
  sg.length = static_cast<uint32_t>(msg_size);
  sg.lkey = mr->lkey;

  ibv_send_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sg;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  ibv_send_wr* bad = nullptr;
  int status = ibv_post_send(qp, &wr, &bad);
  if (status != 0) {
    int err = status < 0 ? -status : status;
    std::cerr << "ibv_post_send failed, status=" << status << " ("
              << std::strerror(err) << ")" << std::endl;
    return false;
  }
  return true;
}

bool poll_completion(ibv_cq* cq, ibv_wc& wc) {
  constexpr int max_polls = 10000000;  // 10M polls max
  for (int polls = 0; polls < max_polls; ++polls) {
    int num = ibv_poll_cq(cq, 1, &wc);
    if (num < 0) {
      std::cerr << "ibv_poll_cq failed" << std::endl;
      return false;
    }
    if (num == 0) {
      if (polls > 0 && polls % 1000000 == 0) {
        std::cerr << "Still polling... (" << polls / 1000000 << "M polls)" << std::endl;
      }
      continue;
    }
    if (wc.status != IBV_WC_SUCCESS) {
      std::cerr << "Completion error: status=" << wc.status
                << " vendor_err=" << wc.vendor_err
                << " opcode=" << wc.opcode << std::endl;
      return false;
    }
    return true;
  }
  std::cerr << "TIMEOUT: No completion after " << max_polls << " polls" << std::endl;
  return false;
}

ibv_device* pick_device(const std::string& requested, ibv_device** list, int num_devices) {
  if (num_devices == 0) {
    return nullptr;
  }
  if (requested.empty()) {
    return list[0];
  }
  for (int i = 0; i < num_devices; ++i) {
    if (requested == ibv_get_device_name(list[i])) {
      return list[i];
    }
  }
  return nullptr;
}

void fill_payload(void* buffer, std::size_t msg_size) {
  uint8_t* ptr = static_cast<uint8_t*>(buffer) + kHeaderSize;
  std::size_t payload_size = msg_size - kHeaderSize;
  for (size_t i = 0; i < payload_size; ++i) {
    ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }
}

// Transition QP to ERROR state to flush pending work requests
bool modify_qp_to_error(ibv_qp* qp) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_ERR;
  int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  return ret == 0;
}

// Transition QP to RESET state for clean destruction
bool modify_qp_to_reset(ibv_qp* qp) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RESET;
  int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  return ret == 0;
}

// Helper to cleanup QPContext resources
void cleanup_qp_context(QPContext& qpc, ibv_pd* pd) {
  (void)pd;  // May be needed for future cleanup

  // Suppress stderr during cleanup to hide Apple driver IOConnectUnmapMemory errors
  StderrSuppressor suppress_stderr;

  // First transition QP to ERROR to flush pending WRs, then to RESET
  if (qpc.qp) {
    // Move to ERROR state - this flushes all pending work requests
    modify_qp_to_error(qpc.qp);

    // Drain any remaining completions from the CQ
    if (qpc.cq) {
      ibv_wc wc{};
      int drain_count = 0;
      while (ibv_poll_cq(qpc.cq, 1, &wc) > 0 && drain_count < 1000) {
        ++drain_count;
      }
    }

    // Move to RESET state for clean destruction
    modify_qp_to_reset(qpc.qp);

    // Destroy QP BEFORE deregistering MRs (Apple driver requirement)
    ibv_destroy_qp(qpc.qp);
    qpc.qp = nullptr;
  }

  // Destroy CQ after QP
  if (qpc.cq) {
    ibv_destroy_cq(qpc.cq);
    qpc.cq = nullptr;
  }

  // Now deregister MRs (must happen after QP destruction, before freeing buffers)
  if (qpc.send_mr) { ibv_dereg_mr(qpc.send_mr); qpc.send_mr = nullptr; }
  if (qpc.recv_mr) { ibv_dereg_mr(qpc.recv_mr); qpc.recv_mr = nullptr; }
  for (auto*& mr : qpc.recv_mrs) {
    if (mr) { ibv_dereg_mr(mr); mr = nullptr; }
  }

  // Free buffers last
  if (qpc.send_buf) { free(qpc.send_buf); qpc.send_buf = nullptr; }
  if (qpc.recv_buf) { free(qpc.recv_buf); qpc.recv_buf = nullptr; }
  for (auto*& buf : qpc.recv_bufs) {
    if (buf) { free(buf); buf = nullptr; }
  }

  qpc = QPContext{};  // Reset to defaults
}

int run_server(const Options& opts) {
  ibv_device** list = nullptr;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  int listen_fd = -1;
  int control_fd = -1;
  int ret = 1;
  std::vector<QPContext> qps;
  int num_devices = 0;

  do {
    list = ibv_get_device_list(&num_devices);
    if (!list || num_devices == 0) {
      std::cerr << "No InfiniBand devices available" << std::endl;
      break;
    }

    ibv_device* device = pick_device(opts.device_name, list, num_devices);
    if (!device) {
      std::cerr << "Requested device not found" << std::endl;
      break;
    }

    ctx = ibv_open_device(device);
    if (!ctx) {
      std::cerr << "Failed to open device" << std::endl;
      break;
    }

    pd = ibv_alloc_pd(ctx);
    if (!pd) {
      std::cerr << "Failed to allocate protection domain" << std::endl;
      break;
    }

    ibv_device_attr dev_attr{};
    if (ibv_query_device(ctx, &dev_attr)) {
      std::cerr << "ibv_query_device failed" << std::endl;
      break;
    }

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, opts.ib_port, &port_attr)) {
      std::cerr << "ibv_query_port failed" << std::endl;
      break;
    }

    // Debug: print all available GIDs
    std::cout << "Available GIDs on port " << static_cast<int>(opts.ib_port) << ":" << std::endl;
    for (int gi = 0; gi < kMaxGIDIndex; ++gi) {
      ibv_gid test_gid{};
      if (ibv_query_gid(ctx, opts.ib_port, gi, &test_gid) == 0) {
        char test_gid_str[64];
        inet_ntop(AF_INET6, test_gid.raw, test_gid_str, sizeof(test_gid_str));
        bool is_v4 = is_ipv4_mapped_gid(test_gid);
        std::cout << "  GID[" << gi << "]: " << test_gid_str
                  << (is_v4 ? " (IPv4-mapped)" : "") << std::endl;
      }
    }

    // Auto-detect valid GID index if needed
    ibv_gid gid{};
    int actual_gid_index = find_valid_gid_index(ctx, opts.ib_port, opts.gid_index, gid);
    if (actual_gid_index < 0) {
      std::cerr << "No valid IPv4-mapped GID found!" << std::endl;
      break;
    }
    // Print selected GID
    {
      char gid_str[64];
      inet_ntop(AF_INET6, gid.raw, gid_str, sizeof(gid_str));
      std::cout << "Using GID[" << actual_gid_index << "]: " << gid_str << std::endl;
    }

    listen_fd = create_listen_socket(opts.listen_port);
    if (listen_fd < 0) {
      break;
    }
    std::cout << "Server waiting on control port " << opts.listen_port
              << " (Ctrl-C to exit)" << std::endl;

    int test_count = 0;
    bool server_running = true;

    while (server_running) {
      control_fd = accept_control(listen_fd);
      if (control_fd < 0) {
        break;
      }
      ++test_count;
      std::cout << "\n=== Test #" << test_count << " ===" << std::endl;

      // First exchange test parameters to know how many QPs to create
      WireTestParams local_params{};
      local_params.iterations = static_cast<uint32_t>(opts.iterations);
      local_params.message_size = static_cast<uint32_t>(opts.message_size);
      local_params.num_qps = static_cast<uint32_t>(opts.num_qps);
      WireTestParams remote_params{};
      if (!exchange_test_params(control_fd, local_params, remote_params, true)) {
        std::cerr << "Failed to exchange test parameters" << std::endl;
        ::close(control_fd);
        control_fd = -1;
        continue;
      }

      // Use client's parameters
      int iterations = static_cast<int>(remote_params.iterations);
      std::size_t msg_size = static_cast<std::size_t>(remote_params.message_size);
      int num_qps = static_cast<int>(remote_params.num_qps);
      if (num_qps < 1 || num_qps > kMaxQPs) num_qps = 1;
      std::cout << "Using client parameters: iterations=" << iterations
                << " message_size=" << msg_size << " num_qps=" << num_qps << std::endl;

      const std::size_t buf_size = std::max(msg_size, static_cast<std::size_t>(4096));
      const int cq_entries = std::max(1, std::min<int>(dev_attr.max_cqe, kRecvWindowSize * 2 + 16));
      const int max_wr = std::max(1, std::min<int>(dev_attr.max_qp_wr, kRecvWindowSize * 2 + 16));
      const int max_sge = std::max(1, std::min<int>(dev_attr.max_sge, 1));
      const int mr_access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

      // Cleanup any previous QPs
      for (auto& qpc : qps) {
        cleanup_qp_context(qpc, pd);
      }
      qps.clear();
      qps.resize(num_qps);

      std::random_device rd;
      bool setup_ok = true;

      // Create CQs, QPs, and allocate buffers for each QP
      for (int q = 0; q < num_qps && setup_ok; ++q) {
        QPContext& qpc = qps[q];
        qpc.psn = rd() & 0xFFFFFF;

        // Create CQ for this QP
        qpc.cq = ibv_create_cq(ctx, cq_entries, nullptr, nullptr, 0);
        if (!qpc.cq) {
          std::cerr << "Failed to create CQ for QP " << q << std::endl;
          setup_ok = false;
          break;
        }

        // Create QP
        ibv_qp_init_attr init_attr{};
        init_attr.qp_context = ctx;
        init_attr.send_cq = qpc.cq;
        init_attr.recv_cq = qpc.cq;
        init_attr.srq = nullptr;
        init_attr.cap.max_send_wr = max_wr;
        init_attr.cap.max_recv_wr = max_wr;
        init_attr.cap.max_send_sge = max_sge;
        init_attr.cap.max_recv_sge = max_sge;
        init_attr.cap.max_inline_data = 0;
        init_attr.qp_type = IBV_QPT_UC;
        init_attr.sq_sig_all = 0;

        qpc.qp = ibv_create_qp(pd, &init_attr);
        if (!qpc.qp) {
          std::cerr << "Failed to create QP " << q << ": " << std::strerror(errno) << std::endl;
          setup_ok = false;
          break;
        }

        if (!modify_qp_to_init(qpc.qp, opts.ib_port)) {
          std::cerr << "Failed to move QP " << q << " to INIT" << std::endl;
          setup_ok = false;
          break;
        }

        // Allocate receive buffers (sliding window) for this QP
        qpc.recv_bufs.resize(kRecvWindowSize, nullptr);
        qpc.recv_mrs.resize(kRecvWindowSize, nullptr);
        for (int i = 0; i < kRecvWindowSize; ++i) {
          qpc.recv_bufs[i] = allocate_page_aligned(buf_size);
          if (!qpc.recv_bufs[i]) {
            std::cerr << "Failed to allocate recv buffer " << i << " for QP " << q << std::endl;
            setup_ok = false;
            break;
          }
          std::memset(qpc.recv_bufs[i], 0, buf_size);
          qpc.recv_mrs[i] = ibv_reg_mr(pd, qpc.recv_bufs[i], buf_size, mr_access);
          if (!qpc.recv_mrs[i]) {
            std::cerr << "Failed to register recv memory " << i << " for QP " << q << std::endl;
            setup_ok = false;
            break;
          }
        }
        if (!setup_ok) break;

        // Allocate send buffer for this QP
        qpc.send_buf = allocate_page_aligned(buf_size);
        if (!qpc.send_buf) {
          std::cerr << "Failed to allocate send buffer for QP " << q << std::endl;
          setup_ok = false;
          break;
        }
        std::memset(qpc.send_buf, 0, buf_size);
        qpc.send_mr = ibv_reg_mr(pd, qpc.send_buf, buf_size, mr_access);
        if (!qpc.send_mr) {
          std::cerr << "Failed to register send memory for QP " << q << std::endl;
          setup_ok = false;
          break;
        }
      }

      if (!setup_ok) {
        for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
        qps.clear();
        ::close(control_fd);
        control_fd = -1;
        continue;
      }

      std::cout << "Created " << num_qps << " QPs with " << kRecvWindowSize
                << " recv buffers each" << std::endl;

      // Exchange QP info for each QP
      std::vector<WireQPInfo> local_qp_infos(num_qps);
      std::vector<WireQPInfo> remote_qp_infos(num_qps);

      for (int q = 0; q < num_qps; ++q) {
        local_qp_infos[q].lid = port_attr.lid;
        local_qp_infos[q].qp_num = qps[q].qp->qp_num;
        local_qp_infos[q].psn = qps[q].psn;
        std::memcpy(local_qp_infos[q].gid, gid.raw, 16);
      }

      // Debug: print local GID being sent
      char gid_str[64];
      inet_ntop(AF_INET6, gid.raw, gid_str, sizeof(gid_str));
      std::cout << "Server local GID: " << gid_str << std::endl;

      // Exchange all QP infos sequentially
      bool exchange_ok = true;
      for (int q = 0; q < num_qps && exchange_ok; ++q) {
        if (!exchange_qp_info(control_fd, local_qp_infos[q], remote_qp_infos[q], true)) {
          std::cerr << "Failed to exchange QP info for QP " << q << std::endl;
          exchange_ok = false;
        }
        // Debug: print received remote GID
        char remote_gid_str[64];
        inet_ntop(AF_INET6, remote_qp_infos[q].gid, remote_gid_str, sizeof(remote_gid_str));
        std::cout << "Server received remote GID[" << q << "]: " << remote_gid_str << std::endl;
      }

      if (!exchange_ok) {
        for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
        qps.clear();
        ::close(control_fd);
        control_fd = -1;
        continue;
      }

      // Transition all QPs to RTR then RTS
      bool transition_ok = true;
      for (int q = 0; q < num_qps && transition_ok; ++q) {
        ibv_gid remote_gid{};
        std::memcpy(remote_gid.raw, remote_qp_infos[q].gid, 16);
        bool use_gid = is_gid_nonzero(remote_gid);

        std::cout << "Transitioning QP " << q << " to RTR/RTS..." << std::endl;
        if (!modify_qp_to_rtr(qps[q].qp, remote_qp_infos[q], opts.ib_port, use_gid, static_cast<uint8_t>(actual_gid_index))) {
          std::cerr << "Failed to move QP " << q << " to RTR" << std::endl;
          transition_ok = false;
          break;
        }
        if (!modify_qp_to_rts(qps[q].qp, qps[q].psn)) {
          std::cerr << "Failed to move QP " << q << " to RTS" << std::endl;
          transition_ok = false;
          break;
        }
      }

      if (!transition_ok) {
        for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
        qps.clear();
        ::close(control_fd);
        control_fd = -1;
        continue;
      }

      // Pre-post receives on all QPs
      // wr_id encoding: (qp_index << 16) | (buf_index + 1)
      bool post_ok = true;
      for (int q = 0; q < num_qps && post_ok; ++q) {
        for (int i = 0; i < kRecvWindowSize; ++i) {
          uint64_t wr_id = (static_cast<uint64_t>(q) << 16) | static_cast<uint64_t>(i + 1);
          if (!post_receive(qps[q].qp, qps[q].recv_bufs[i], qps[q].recv_mrs[i], wr_id, msg_size)) {
            std::cerr << "Failed to post recv " << i << " on QP " << q << std::endl;
            post_ok = false;
            break;
          }
        }
      }

      if (!post_ok) {
        for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
        qps.clear();
        ::close(control_fd);
        control_fd = -1;
        continue;
      }

      std::cout << "Server ready, " << num_qps << " QPs in RTS state, waiting for "
                << iterations << " messages..." << std::endl;

      // Synchronization barrier
      uint8_t ready = 1;
      if (!send_all(control_fd, &ready, 1) || !recv_all(control_fd, &ready, 1)) {
        std::cerr << "Failed to synchronize ready state" << std::endl;
        for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
        qps.clear();
        ::close(control_fd);
        control_fd = -1;
        continue;
      }
      std::cout << "Synchronized with client, ready to receive" << std::endl;

      // Main receive/echo loop - poll all CQs round-robin
      uint64_t test_start_ns = wall_time_ns();
      int total_handled = 0;
      const int max_outstanding_per_qp = max_wr / 2;
      bool success = true;
      int progress_interval = iterations / 10;
      if (progress_interval < 1) progress_interval = 1;
      int last_progress = -1;

      std::cout << "Starting main loop, waiting for messages..." << std::endl;

      // Idle timeout: if no messages received for 2 seconds, assume client is done
      uint64_t last_activity_ns = wall_time_ns();
      constexpr uint64_t kIdleTimeoutNs = 2000000000ULL;  // 2 seconds

      while (total_handled < iterations && success) {
        // Progress output (only when progress changes)
        int current_progress = total_handled / progress_interval;
        if (current_progress > last_progress && total_handled > 0) {
          std::cout << "  Server handled " << total_handled << "/" << iterations << " messages" << std::endl;
          last_progress = current_progress;
        }

        // Check idle timeout
        uint64_t now_ns = wall_time_ns();
        if (total_handled > 0 && (now_ns - last_activity_ns) > kIdleTimeoutNs) {
          std::cout << "Idle timeout after " << total_handled << "/" << iterations
                    << " messages (no activity for 2s)" << std::endl;
          break;
        }

        // Poll each CQ in round-robin
        for (int q = 0; q < num_qps && total_handled < iterations; ++q) {
          QPContext& qpc = qps[q];
          ibv_wc wc{};
          int num = ibv_poll_cq(qpc.cq, 1, &wc);

          if (num < 0) {
            std::cerr << "ibv_poll_cq failed on QP " << q << std::endl;
            success = false;
            break;
          }

          if (num == 0) continue;

          if (wc.status != IBV_WC_SUCCESS) {
            std::cerr << "Completion error on QP " << q << ": status=" << wc.status << std::endl;
            success = false;
            break;
          }

          // Decode wr_id: (qp_index << 16) | (buf_index + 1)
          int wc_qp = static_cast<int>(wc.wr_id >> 16);
          int buf_idx = static_cast<int>(wc.wr_id & 0xFFFF) - 1;

          if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
            if (wc_qp != q || buf_idx < 0 || buf_idx >= kRecvWindowSize) {
              std::cerr << "Invalid recv wr_id on QP " << q << std::endl;
              success = false;
              break;
            }

            // Drain sends if needed
            while (qpc.outstanding_sends >= max_outstanding_per_qp) {
              ibv_wc send_wc{};
              int snum = ibv_poll_cq(qpc.cq, 1, &send_wc);
              if (snum < 0) {
                success = false;
                break;
              }
              if (snum > 0 && send_wc.opcode == IBV_WC_SEND) {
                --qpc.outstanding_sends;
              }
            }
            if (!success) break;

            // Echo back on same QP
            std::memcpy(qpc.send_buf, qpc.recv_bufs[buf_idx], msg_size);
            // Send wr_id: high bit set + (qp << 16) + handled count
            uint64_t send_wr_id = (1ULL << 31) | (static_cast<uint64_t>(q) << 16) | static_cast<uint64_t>(qpc.handled);
            if (!post_send(qpc.qp, qpc.send_buf, qpc.send_mr, send_wr_id, msg_size)) {
              std::cerr << "Failed to post echo on QP " << q << std::endl;
              success = false;
              break;
            }
            ++qpc.outstanding_sends;
            ++qpc.handled;
            ++total_handled;
            last_activity_ns = wall_time_ns();  // Reset idle timeout

            // Re-post receive
            uint64_t new_wr_id = (static_cast<uint64_t>(q) << 16) | static_cast<uint64_t>(buf_idx + 1);
            if (!post_receive(qpc.qp, qpc.recv_bufs[buf_idx], qpc.recv_mrs[buf_idx], new_wr_id, msg_size)) {
              success = false;
              break;
            }
          } else if (wc.opcode == IBV_WC_SEND) {
            --qpc.outstanding_sends;
          }
        }
      }

      // Drain outstanding sends on all QPs
      for (int q = 0; q < num_qps && success; ++q) {
        QPContext& qpc = qps[q];
        while (qpc.outstanding_sends > 0) {
          ibv_wc wc{};
          int num = ibv_poll_cq(qpc.cq, 1, &wc);
          if (num < 0) {
            success = false;
            break;
          }
          if (num > 0 && wc.opcode == IBV_WC_SEND) {
            --qpc.outstanding_sends;
          }
        }
      }

      uint64_t test_end_ns = wall_time_ns();
      double test_duration_s = static_cast<double>(test_end_ns - test_start_ns) / 1e9;
      uint64_t total_bytes = static_cast<uint64_t>(total_handled) * 2 * msg_size;
      double bandwidth_gbps = static_cast<double>(total_bytes) / test_duration_s / 1e9;
      double msg_rate = static_cast<double>(total_handled) / test_duration_s;

      if (success && total_handled == iterations) {
        std::cout << "\n=== Server Test #" << test_count << " Complete ===" << std::endl;
        std::cout << "QPs used:         " << num_qps << std::endl;
        std::cout << "Iterations:       " << total_handled << std::endl;
        for (int q = 0; q < num_qps; ++q) {
          std::cout << "  QP " << q << " handled:  " << qps[q].handled << std::endl;
        }
        std::cout << "Message size:     " << msg_size << " bytes" << std::endl;
        std::cout << "Test duration:    " << std::fixed << std::setprecision(3) << test_duration_s << " s" << std::endl;
        std::cout << "Bandwidth:        " << std::fixed << std::setprecision(4) << bandwidth_gbps << " GB/s" << std::endl;
        std::cout << "Message rate:     " << std::fixed << std::setprecision(0) << msg_rate << " msg/s" << std::endl;
        ret = 0;
      } else {
        std::cout << "Test #" << test_count << " failed after " << total_handled << " iterations" << std::endl;
      }

      // Cleanup QPs for this test (will recreate for next client)
      std::cout << "Cleaning up " << qps.size() << " QPs..." << std::endl;
      for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
      qps.clear();
      std::cout << "QP cleanup complete" << std::endl;

      ::close(control_fd);
      control_fd = -1;

      // Full device restart to clear hardware state (Apple driver quirk)
      std::cout << "Restarting RDMA device to clear hardware state..." << std::endl;
      {
        StderrSuppressor suppress_stderr;
        if (pd) { ibv_dealloc_pd(pd); pd = nullptr; }
        if (ctx) { ibv_close_device(ctx); ctx = nullptr; }
      }
      // Also free and re-get device list to fully reset driver state
      if (list) { ibv_free_device_list(list); list = nullptr; }
      usleep(500000);  // 500ms delay for hardware to fully reset

      // Re-get device list
      list = ibv_get_device_list(&num_devices);
      if (!list || num_devices == 0) {
        std::cerr << "Failed to get device list on restart" << std::endl;
        server_running = false;
        break;
      }

      // Reopen device
      ibv_device* device = pick_device(opts.device_name, list, num_devices);
      if (!device) {
        std::cerr << "Failed to find device on restart" << std::endl;
        server_running = false;
        break;
      }
      ctx = ibv_open_device(device);
      if (!ctx) {
        std::cerr << "Failed to reopen device" << std::endl;
        server_running = false;
        break;
      }
      pd = ibv_alloc_pd(ctx);
      if (!pd) {
        std::cerr << "Failed to reallocate PD" << std::endl;
        server_running = false;
        break;
      }
      // Re-query device attributes
      if (ibv_query_device(ctx, &dev_attr)) {
        std::cerr << "ibv_query_device failed on restart" << std::endl;
        server_running = false;
        break;
      }
      if (ibv_query_port(ctx, opts.ib_port, &port_attr)) {
        std::cerr << "ibv_query_port failed on restart" << std::endl;
        server_running = false;
        break;
      }
      // Re-detect GID
      actual_gid_index = find_valid_gid_index(ctx, opts.ib_port, opts.gid_index, gid);
      if (actual_gid_index < 0) {
        std::cerr << "No valid GID found on restart" << std::endl;
        server_running = false;
        break;
      }

      std::cout << "Device restarted successfully" << std::endl;
      std::cout << "\nWaiting for next client (Ctrl-C to exit)..." << std::endl;
    }
  } while (false);

  // Final cleanup
  for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
  if (control_fd >= 0) ::close(control_fd);
  if (listen_fd >= 0) ::close(listen_fd);
  {
    StderrSuppressor suppress_stderr;
    if (pd) ibv_dealloc_pd(pd);
    if (ctx) ibv_close_device(ctx);
  }
  if (list) ibv_free_device_list(list);
  return ret;
}

int run_client(const Options& opts) {
  ibv_device** list = nullptr;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  int control_fd = -1;
  int ret = 1;
  std::vector<QPContext> qps;
  const int num_qps = opts.num_qps;
  const std::size_t msg_size = opts.message_size;
  const std::size_t buf_size = std::max(msg_size, static_cast<std::size_t>(4096));

  do {
    int num_devices = 0;
    list = ibv_get_device_list(&num_devices);
    if (!list || num_devices == 0) {
      std::cerr << "No InfiniBand devices available" << std::endl;
      break;
    }

    ibv_device* device = pick_device(opts.device_name, list, num_devices);
    if (!device) {
      std::cerr << "Requested device not found" << std::endl;
      break;
    }

    ctx = ibv_open_device(device);
    if (!ctx) {
      std::cerr << "Failed to open device" << std::endl;
      break;
    }

    pd = ibv_alloc_pd(ctx);
    if (!pd) {
      std::cerr << "Failed to allocate protection domain" << std::endl;
      break;
    }

    ibv_device_attr dev_attr{};
    if (ibv_query_device(ctx, &dev_attr)) {
      std::cerr << "ibv_query_device failed" << std::endl;
      break;
    }

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, opts.ib_port, &port_attr)) {
      std::cerr << "ibv_query_port failed" << std::endl;
      break;
    }

    // Debug: print all available GIDs
    std::cout << "Available GIDs on port " << static_cast<int>(opts.ib_port) << ":" << std::endl;
    for (int gi = 0; gi < kMaxGIDIndex; ++gi) {
      ibv_gid test_gid{};
      if (ibv_query_gid(ctx, opts.ib_port, gi, &test_gid) == 0) {
        char test_gid_str[64];
        inet_ntop(AF_INET6, test_gid.raw, test_gid_str, sizeof(test_gid_str));
        bool is_v4 = is_ipv4_mapped_gid(test_gid);
        std::cout << "  GID[" << gi << "]: " << test_gid_str
                  << (is_v4 ? " (IPv4-mapped)" : "") << std::endl;
      }
    }

    // Auto-detect valid GID index if needed
    ibv_gid gid{};
    int actual_gid_index = find_valid_gid_index(ctx, opts.ib_port, opts.gid_index, gid);
    if (actual_gid_index < 0) {
      std::cerr << "No valid IPv4-mapped GID found!" << std::endl;
      break;
    }
    // Print selected GID
    {
      char gid_str[64];
      inet_ntop(AF_INET6, gid.raw, gid_str, sizeof(gid_str));
      std::cout << "Using GID[" << actual_gid_index << "]: " << gid_str << std::endl;
    }

    control_fd = connect_control(opts.connect_host, opts.connect_port);
    if (control_fd < 0) {
      break;
    }

    // Exchange test parameters first
    WireTestParams local_params{};
    local_params.iterations = static_cast<uint32_t>(opts.iterations);
    local_params.message_size = static_cast<uint32_t>(opts.message_size);
    local_params.num_qps = static_cast<uint32_t>(num_qps);
    WireTestParams remote_params{};
    if (!exchange_test_params(control_fd, local_params, remote_params, false)) {
      std::cerr << "Failed to exchange test parameters" << std::endl;
      break;
    }
    std::cout << "Sent test parameters to server: iterations=" << opts.iterations
              << " message_size=" << msg_size << " num_qps=" << num_qps << std::endl;

    // Create CQs, QPs, and buffers for each QP
    const int cq_entries = std::max(1, std::min<int>(dev_attr.max_cqe, 256));
    const int max_wr = std::max(1, std::min<int>(dev_attr.max_qp_wr, 64));
    const int max_sge = std::max(1, std::min<int>(dev_attr.max_sge, 1));
    const int mr_access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

    qps.resize(num_qps);
    std::random_device rd;
    bool setup_ok = true;

    for (int q = 0; q < num_qps && setup_ok; ++q) {
      QPContext& qpc = qps[q];
      qpc.psn = rd() & 0xFFFFFF;

      // Create CQ
      qpc.cq = ibv_create_cq(ctx, cq_entries, nullptr, nullptr, 0);
      if (!qpc.cq) {
        std::cerr << "Failed to create CQ for QP " << q << std::endl;
        setup_ok = false;
        break;
      }

      // Create QP
      ibv_qp_init_attr init_attr{};
      init_attr.qp_context = ctx;
      init_attr.send_cq = qpc.cq;
      init_attr.recv_cq = qpc.cq;
      init_attr.srq = nullptr;
      init_attr.cap.max_send_wr = max_wr;
      init_attr.cap.max_recv_wr = max_wr;
      init_attr.cap.max_send_sge = max_sge;
      init_attr.cap.max_recv_sge = max_sge;
      init_attr.cap.max_inline_data = 0;
      init_attr.qp_type = IBV_QPT_UC;
      init_attr.sq_sig_all = 0;

      qpc.qp = ibv_create_qp(pd, &init_attr);
      if (!qpc.qp) {
        std::cerr << "Failed to create QP " << q << ": " << std::strerror(errno) << std::endl;
        setup_ok = false;
        break;
      }

      if (!modify_qp_to_init(qpc.qp, opts.ib_port)) {
        std::cerr << "Failed to move QP " << q << " to INIT" << std::endl;
        setup_ok = false;
        break;
      }

      // Allocate buffers
      qpc.send_buf = allocate_page_aligned(buf_size);
      qpc.recv_buf = allocate_page_aligned(buf_size);
      if (!qpc.send_buf || !qpc.recv_buf) {
        std::cerr << "Failed to allocate buffers for QP " << q << std::endl;
        setup_ok = false;
        break;
      }
      std::memset(qpc.send_buf, 0, buf_size);
      std::memset(qpc.recv_buf, 0, buf_size);

      qpc.send_mr = ibv_reg_mr(pd, qpc.send_buf, buf_size, mr_access);
      qpc.recv_mr = ibv_reg_mr(pd, qpc.recv_buf, buf_size, mr_access);
      if (!qpc.send_mr || !qpc.recv_mr) {
        std::cerr << "Failed to register memory for QP " << q << std::endl;
        setup_ok = false;
        break;
      }
    }

    if (!setup_ok) {
      for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
      break;
    }

    std::cout << "Created " << num_qps << " QPs with CQ size " << cq_entries << std::endl;

    // Exchange QP info for each QP
    std::vector<WireQPInfo> local_qp_infos(num_qps);
    std::vector<WireQPInfo> remote_qp_infos(num_qps);

    for (int q = 0; q < num_qps; ++q) {
      local_qp_infos[q].lid = port_attr.lid;
      local_qp_infos[q].qp_num = qps[q].qp->qp_num;
      local_qp_infos[q].psn = qps[q].psn;
      std::memcpy(local_qp_infos[q].gid, gid.raw, 16);
    }

    // Debug: print local GID being sent
    char gid_str[64];
    inet_ntop(AF_INET6, gid.raw, gid_str, sizeof(gid_str));
    std::cout << "Client local GID: " << gid_str << std::endl;

    bool exchange_ok = true;
    for (int q = 0; q < num_qps && exchange_ok; ++q) {
      if (!exchange_qp_info(control_fd, local_qp_infos[q], remote_qp_infos[q], false)) {
        std::cerr << "Failed to exchange QP info for QP " << q << std::endl;
        exchange_ok = false;
      }
      // Debug: print received remote GID
      char remote_gid_str[64];
      inet_ntop(AF_INET6, remote_qp_infos[q].gid, remote_gid_str, sizeof(remote_gid_str));
      std::cout << "Client received remote GID[" << q << "]: " << remote_gid_str << std::endl;
    }

    if (!exchange_ok) {
      for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
      break;
    }

    // Transition all QPs to RTR then RTS
    bool transition_ok = true;
    for (int q = 0; q < num_qps && transition_ok; ++q) {
      ibv_gid remote_gid{};
      std::memcpy(remote_gid.raw, remote_qp_infos[q].gid, 16);
      bool use_gid = is_gid_nonzero(remote_gid);

      if (!modify_qp_to_rtr(qps[q].qp, remote_qp_infos[q], opts.ib_port, use_gid, static_cast<uint8_t>(actual_gid_index))) {
        std::cerr << "Failed to move QP " << q << " to RTR" << std::endl;
        transition_ok = false;
        break;
      }
      if (!modify_qp_to_rts(qps[q].qp, qps[q].psn)) {
        std::cerr << "Failed to move QP " << q << " to RTS" << std::endl;
        transition_ok = false;
        break;
      }
    }

    if (!transition_ok) {
      for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
      break;
    }

    // Post initial receives on all QPs
    // wr_id encoding: (qp_index << 16) | 1 (recv ID)
    for (int q = 0; q < num_qps; ++q) {
      uint64_t wr_id = (static_cast<uint64_t>(q) << 16) | 1;
      if (!post_receive(qps[q].qp, qps[q].recv_buf, qps[q].recv_mr, wr_id, msg_size)) {
        std::cerr << "Failed to post initial receive on QP " << q << std::endl;
        transition_ok = false;
        break;
      }
    }

    if (!transition_ok) {
      for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
      break;
    }

    std::cout << "Client ready, " << num_qps << " QPs in RTS state" << std::endl;

    // Synchronization barrier
    uint8_t ready = 1;
    if (!send_all(control_fd, &ready, 1) || !recv_all(control_fd, &ready, 1)) {
      std::cerr << "Failed to synchronize ready state" << std::endl;
      for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
      break;
    }
    std::cout << "Synchronized with server, starting to send" << std::endl;

    // Fill payload for all QPs
    for (int q = 0; q < num_qps; ++q) {
      fill_payload(qps[q].send_buf, msg_size);
    }

    bool success = true;
    std::vector<double> latencies;
    constexpr int warmup_iters = 100;
    std::vector<uint64_t> received_seqs;
    uint64_t total_sent = 0;
    uint64_t total_received = 0;

    uint64_t test_start_ns = wall_time_ns();

    if (opts.flood_outstanding >= 0) {
      // Flood mode with multiple QPs - distribute sends round-robin
      // --flood N means N packets PER QP, so total = N * num_qps
      const int per_qp_outstanding = (opts.flood_outstanding == 0) ? (max_wr - 2) :
                                      std::max(1, opts.flood_outstanding);
      const int max_outstanding_total = per_qp_outstanding * num_qps;
      // Per-QP limit to avoid overflowing any single QP's send queue
      const int max_outstanding_per_qp = std::min(per_qp_outstanding, max_wr - 2);
      std::cout << "FLOOD MODE: " << num_qps << " QPs, " << per_qp_outstanding
                << " outstanding/QP, " << max_outstanding_total << " total" << std::endl;

      int progress_interval = opts.iterations / 10;
      if (progress_interval < 1) progress_interval = 1;

      // Track total outstanding across all QPs
      int total_outstanding = 0;

      for (int iter = 0; iter < opts.iterations && success; ++iter) {
        if (iter > 0 && iter % progress_interval == 0) {
          std::cout << "  Sent " << iter << "/" << opts.iterations << " messages" << std::endl;
        }

        // Select QP in round-robin
        int q = iter % num_qps;
        QPContext& qpc = qps[q];

        // Poll ALL CQs to free up send queues and receive echoes
        int poll_attempts = 0;
        while (total_outstanding >= max_outstanding_total || qpc.outstanding_sends >= max_outstanding_per_qp) {
          bool found_any = false;
          for (int pq = 0; pq < num_qps; ++pq) {
            QPContext& poll_qpc = qps[pq];
            ibv_wc wc{};
            int num = ibv_poll_cq(poll_qpc.cq, 1, &wc);
            if (num < 0) {
              std::cerr << "ibv_poll_cq failed on QP " << pq << std::endl;
              success = false;
              break;
            }
            if (num > 0) {
              found_any = true;
              if (wc.status != IBV_WC_SUCCESS) {
                std::cerr << "Completion error on QP " << pq << ": status=" << wc.status << std::endl;
                success = false;
                break;
              }
              if (wc.opcode == IBV_WC_SEND) {
                --poll_qpc.outstanding_sends;
                --total_outstanding;
              } else if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                uint64_t now_ns = wall_time_ns();
                MessageHeader* recv_hdr = static_cast<MessageHeader*>(poll_qpc.recv_buf);
                double latency_us = static_cast<double>(now_ns - recv_hdr->send_time_ns) / 1000.0;
                received_seqs.push_back(recv_hdr->sequence);
                ++total_received;
                if (received_seqs.size() > static_cast<size_t>(warmup_iters)) {
                  latencies.push_back(latency_us);
                }
                uint64_t wr_id = (static_cast<uint64_t>(pq) << 16) | 1;
                if (!post_receive(poll_qpc.qp, poll_qpc.recv_buf, poll_qpc.recv_mr, wr_id, msg_size)) {
                  std::cerr << "Failed to repost receive on QP " << pq << std::endl;
                }
              }
            }
          }
          if (!success) break;
          ++poll_attempts;
          if (poll_attempts > 10000000) {
            std::cerr << "TIMEOUT waiting (total_outstanding=" << total_outstanding
                      << ", qp" << q << "_outstanding=" << qpc.outstanding_sends << ")" << std::endl;
            success = false;
            break;
          }
        }
        if (!success) break;

        // Send on this QP
        ++qpc.sequence;
        ++total_sent;
        MessageHeader* send_hdr = static_cast<MessageHeader*>(qpc.send_buf);
        send_hdr->sequence = total_sent;  // Use global sequence for tracking
        send_hdr->send_time_ns = wall_time_ns();

        // Send wr_id: (qp << 16) | 2
        uint64_t send_wr_id = (static_cast<uint64_t>(q) << 16) | 2;
        if (!post_send(qpc.qp, qpc.send_buf, qpc.send_mr, send_wr_id, msg_size)) {
          std::cerr << "Failed to post send on QP " << q << std::endl;
          success = false;
          break;
        }
        ++qpc.outstanding_sends;
        ++total_outstanding;
      }

      // Drain all outstanding sends and wait for echoes - poll ALL CQs together
      std::cout << "Draining outstanding (total=" << total_outstanding << ")..." << std::endl;
      int drain_timeout = 0;
      while ((total_outstanding > 0 || total_received < total_sent) && drain_timeout < 10000000 && success) {
        bool found_any = false;
        for (int q = 0; q < num_qps && success; ++q) {
          QPContext& qpc = qps[q];
          ibv_wc wc{};
          int num = ibv_poll_cq(qpc.cq, 1, &wc);
          if (num < 0) {
            success = false;
            break;
          }
          if (num > 0) {
            found_any = true;
            if (wc.status != IBV_WC_SUCCESS) {
              std::cerr << "Drain: completion error on QP " << q << ": status=" << wc.status << std::endl;
              // Don't fail, just continue draining
            }
            if (wc.opcode == IBV_WC_SEND) {
              --qpc.outstanding_sends;
              --total_outstanding;
            } else if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              uint64_t now_ns = wall_time_ns();
              MessageHeader* recv_hdr = static_cast<MessageHeader*>(qpc.recv_buf);
              double latency_us = static_cast<double>(now_ns - recv_hdr->send_time_ns) / 1000.0;
              received_seqs.push_back(recv_hdr->sequence);
              ++total_received;
              if (received_seqs.size() > static_cast<size_t>(warmup_iters)) {
                latencies.push_back(latency_us);
              }
              uint64_t wr_id = (static_cast<uint64_t>(q) << 16) | 1;
              post_receive(qpc.qp, qpc.recv_buf, qpc.recv_mr, wr_id, msg_size);
            }
          }
        }
        if (!found_any) {
          ++drain_timeout;
        } else {
          drain_timeout = 0;  // Reset timeout when we find completions
        }
      }
      if (drain_timeout >= 10000000) {
        std::cout << "Drain timeout (outstanding=" << total_outstanding
                  << ", received=" << total_received << "/" << total_sent << ")" << std::endl;
      }
      std::cout << "Flood complete: sent " << total_sent << ", received "
                << total_received << " echoes" << std::endl;

    } else {
      // Ping-pong mode with multiple QPs - still round-robin but wait for each echo
      for (int iter = 0; iter < opts.iterations && success; ++iter) {
        int q = iter % num_qps;
        QPContext& qpc = qps[q];

        ++qpc.sequence;
        ++total_sent;
        MessageHeader* send_hdr = static_cast<MessageHeader*>(qpc.send_buf);
        send_hdr->sequence = total_sent;
        send_hdr->send_time_ns = wall_time_ns();

        uint64_t send_wr_id = (static_cast<uint64_t>(q) << 16) | 2;
        if (!post_send(qpc.qp, qpc.send_buf, qpc.send_mr, send_wr_id, msg_size)) {
          std::cerr << "Failed to post send on QP " << q << std::endl;
          success = false;
          break;
        }

        // Wait for send and recv completion on this QP
        bool have_send = false;
        bool have_recv = false;
        while ((!have_send || !have_recv) && success) {
          ibv_wc wc{};
          if (!poll_completion(qpc.cq, wc)) {
            success = false;
            break;
          }
          if (wc.opcode == IBV_WC_SEND) {
            have_send = true;
          } else if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
            have_recv = true;
          }
        }

        if (!success) break;

        uint64_t now_ns = wall_time_ns();
        MessageHeader* recv_hdr = static_cast<MessageHeader*>(qpc.recv_buf);
        double latency_us = static_cast<double>(now_ns - recv_hdr->send_time_ns) / 1000.0;
        received_seqs.push_back(recv_hdr->sequence);
        ++total_received;

        if (iter >= warmup_iters) {
          latencies.push_back(latency_us);
        }

        uint64_t recv_wr_id = (static_cast<uint64_t>(q) << 16) | 1;
        if (!post_receive(qpc.qp, qpc.recv_buf, qpc.recv_mr, recv_wr_id, msg_size)) {
          success = false;
          break;
        }
      }
    }

    if (success && total_sent == static_cast<uint64_t>(opts.iterations)) {
      ret = 0;

      if (!latencies.empty()) {
        double sum = 0.0;
        double min_lat = latencies[0];
        double max_lat = latencies[0];
        for (double lat : latencies) {
          sum += lat;
          if (lat < min_lat) min_lat = lat;
          if (lat > max_lat) max_lat = lat;
        }
        double avg_lat = sum / latencies.size();

        std::cout << "\n=== Latency Statistics (excluding first " << warmup_iters
                  << " warmup iterations) ===" << std::endl;
        std::cout << "Total iterations: " << opts.iterations << std::endl;
        std::cout << "Measured samples: " << latencies.size() << std::endl;
        std::cout << "Min:              " << std::fixed << std::setprecision(2) << min_lat << " us" << std::endl;
        std::cout << "Avg:              " << std::fixed << std::setprecision(2) << avg_lat << " us" << std::endl;
        std::cout << "Max:              " << std::fixed << std::setprecision(2) << max_lat << " us" << std::endl;
      }

      // Per-QP statistics
      std::cout << "\n=== Per-QP Statistics ===" << std::endl;
      std::cout << "QPs used:         " << num_qps << std::endl;
      for (int q = 0; q < num_qps; ++q) {
        std::cout << "  QP " << q << " sent:     " << qps[q].sequence << std::endl;
      }

      // Packet loss statistics
      std::cout << "\n=== Packet Loss Statistics ===" << std::endl;
      std::cout << "Total sent:       " << total_sent << std::endl;
      std::cout << "Total received:   " << total_received << std::endl;
      uint64_t lost = total_sent > total_received ? total_sent - total_received : 0;
      std::cout << "Lost packets:     " << lost << std::endl;
      if (total_sent > 0) {
        double loss_rate = 100.0 * static_cast<double>(lost) / static_cast<double>(total_sent);
        std::cout << "Loss rate:        " << std::fixed << std::setprecision(4) << loss_rate << " %" << std::endl;
      }

      // Bandwidth statistics
      uint64_t test_end_ns = wall_time_ns();
      double test_duration_s = static_cast<double>(test_end_ns - test_start_ns) / 1e9;
      uint64_t total_bytes = total_sent * 2 * msg_size;
      double bandwidth_gbps = static_cast<double>(total_bytes) / test_duration_s / 1e9;
      double msg_rate = static_cast<double>(total_sent) / test_duration_s;

      std::cout << "\n=== Bandwidth Statistics ===" << std::endl;
      std::cout << "Test duration:    " << std::fixed << std::setprecision(3) << test_duration_s << " s" << std::endl;
      std::cout << "Message size:     " << msg_size << " bytes" << std::endl;
      std::cout << "Total data:       " << std::fixed << std::setprecision(2)
                << static_cast<double>(total_bytes) / 1e9 << " GB (send+recv)" << std::endl;
      std::cout << "Bandwidth:        " << std::fixed << std::setprecision(4) << bandwidth_gbps << " GB/s" << std::endl;
      std::cout << "Message rate:     " << std::fixed << std::setprecision(0) << msg_rate << " msg/s" << std::endl;
    }
  } while (false);

  // Cleanup - order matters for Apple driver!
  // 1. First cleanup all QPs (destroys QPs, CQs, deregisters MRs, frees buffers)
  for (auto& qpc : qps) cleanup_qp_context(qpc, pd);
  qps.clear();

  // 2. Close control socket
  if (control_fd >= 0) ::close(control_fd);

  // 3. Small delay to let driver finish internal cleanup
  usleep(10000);  // 10ms

  // 4. Deallocate PD and close device (suppress Apple driver errors)
  {
    StderrSuppressor suppress_stderr;
    if (pd) { ibv_dealloc_pd(pd); pd = nullptr; }
    if (ctx) { ibv_close_device(ctx); ctx = nullptr; }
  }

  // 5. Free device list
  if (list) { ibv_free_device_list(list); list = nullptr; }

  return ret;
}

void usage(const char* prog) {
  std::cerr << "Usage:" << std::endl;
  std::cerr << "  " << prog
            << " --server --listen <port> [--device <name>] [--ib-port <n>]"
            << " [--gid-index <n>]" << std::endl;
  std::cerr << "  " << prog
            << " --client --connect <host:port> [--device <name>] [--ib-port <n>]"
            << " [--gid-index <n>] [--iterations <n>] [--message-size <bytes>] [--flood <n>] [--qp <n>]" << std::endl;
  std::cerr << "  Default message size: " << kDefaultMessageSize << " bytes, max: "
            << kMaxMessageSize << " bytes" << std::endl;
  std::cerr << "  --flood [n]: Flood mode (0 or omit = unlimited, n = limit outstanding)" << std::endl;
  std::cerr << "  --qp <n>: Number of queue pairs (1-" << kMaxQPs << ", default: 1)" << std::endl;
  std::cerr << "  Note: Server uses client's iteration count, message size, and QP count" << std::endl;
}

bool parse_args(int argc, char** argv, Options& opts) {
  if (argc < 2) {
    std::cout << "ibv_roundtrip " << kBuildVersion << std::endl;
    usage(argv[0]);
    return false;
  }

  bool mode_set = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--server") {
      opts.mode = Options::Mode::Server;
      mode_set = true;
    } else if (arg == "--client") {
      opts.mode = Options::Mode::Client;
      mode_set = true;
    } else if (arg == "--device" && i + 1 < argc) {
      opts.device_name = argv[++i];
    } else if (arg == "--ib-port" && i + 1 < argc) {
      opts.ib_port = static_cast<uint8_t>(std::stoi(argv[++i]));
    } else if (arg == "--gid-index" && i + 1 < argc) {
      opts.gid_index = static_cast<uint8_t>(std::stoi(argv[++i]));
    } else if (arg == "--listen" && i + 1 < argc) {
      opts.listen_port = static_cast<uint16_t>(parse_port(argv[++i]));
    } else if (arg == "--connect" && i + 1 < argc) {
      std::string target = argv[++i];
      auto pos = target.find(':');
      if (pos == std::string::npos) {
        throw std::runtime_error("--connect requires host:port");
      }
      opts.connect_host = target.substr(0, pos);
      opts.connect_port = static_cast<uint16_t>(parse_port(target.substr(pos + 1)));
    } else if (arg == "--iterations" && i + 1 < argc) {
      opts.iterations = std::stoi(argv[++i]);
    } else if (arg == "--message-size" && i + 1 < argc) {
      opts.message_size = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (arg == "--flood") {
      // Check if next arg is a number
      if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
        opts.flood_outstanding = std::stoi(argv[++i]);
        // 0 means unlimited, any positive value limits outstanding
      } else {
        opts.flood_outstanding = 0;  // Default to unlimited if no value given
      }
    } else if (arg == "--qp" && i + 1 < argc) {
      opts.num_qps = std::stoi(argv[++i]);
    } else {
      usage(argv[0]);
      return false;
    }
  }

  if (!mode_set) {
    usage(argv[0]);
    return false;
  }

  if (opts.iterations <= 0) {
    throw std::runtime_error("Iterations must be positive");
  }

  if (opts.message_size < kHeaderSize) {
    throw std::runtime_error("Message size must be at least " + std::to_string(kHeaderSize) + " bytes");
  }
  if (opts.message_size > kMaxMessageSize) {
    throw std::runtime_error("Message size must be at most " + std::to_string(kMaxMessageSize) + " bytes");
  }

  if (opts.num_qps < 1 || opts.num_qps > kMaxQPs) {
    throw std::runtime_error("Number of QPs must be between 1 and " + std::to_string(kMaxQPs));
  }

  if (opts.mode == Options::Mode::Server) {
    if (opts.listen_port == 0) {
      throw std::runtime_error("Server mode requires --listen <port>");
    }
  } else {
    if (opts.connect_host.empty()) {
      throw std::runtime_error("Client mode requires --connect host:port");
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options opts{};
    if (!parse_args(argc, argv, opts)) {
      return 1;
    }
    std::cout << (opts.mode == Options::Mode::Server ? "Server" : "Client")
              << " build " << kBuildVersion << std::endl;
    if (opts.mode == Options::Mode::Server) {
      return run_server(opts);
    }
    return run_client(opts);
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }
}
