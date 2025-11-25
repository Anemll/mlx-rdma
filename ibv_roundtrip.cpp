#include <arpa/inet.h>
#include <errno.h>
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

constexpr std::size_t kDefaultMessageSize = 1000;
constexpr std::size_t kMaxMessageSize = 65536;  // 64KB max
constexpr const char* kBuildVersion = "v0.0.37";

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
  int flood_outstanding = 0;  // 0 = disabled, >0 = flood mode with N outstanding
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
  clock_gettime(CLOCK_REALTIME, &ts);
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

int run_server(const Options& opts) {
  ibv_device** list = nullptr;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  std::vector<void*> recv_bufs(kRecvWindowSize, nullptr);
  void* send_buf = nullptr;
  std::vector<ibv_mr*> recv_mrs(kRecvWindowSize, nullptr);
  ibv_mr* send_mr = nullptr;
  int listen_fd = -1;
  int control_fd = -1;
  int ret = 1;
  std::size_t msg_size = opts.message_size;  // Will be overridden by client
  int iterations = opts.iterations;           // Will be overridden by client

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

    // Increase CQ size to handle window of receives + sends
    int cq_entries = std::max(1, std::min<int>(dev_attr.max_cqe, kRecvWindowSize * 2 + 16));
    cq = ibv_create_cq(ctx, cq_entries, nullptr, nullptr, 0);
    if (!cq) {
      std::cerr << "Failed to create completion queue" << std::endl;
      break;
    }

    ibv_qp_init_attr init_attr{};
    init_attr.qp_context = ctx;
    init_attr.send_cq = cq;
    init_attr.recv_cq = cq;
    init_attr.srq = nullptr;
    // Increase QP capacity to handle sliding window
    int max_wr = std::max(1, std::min<int>(dev_attr.max_qp_wr, kRecvWindowSize * 2 + 16));
    int max_sge = std::max(1, std::min<int>(dev_attr.max_sge, 1));
    init_attr.cap.max_send_wr = max_wr;
    init_attr.cap.max_recv_wr = max_wr;
    init_attr.cap.max_send_sge = max_sge;
    init_attr.cap.max_recv_sge = max_sge;
    init_attr.cap.max_inline_data = 0;
    init_attr.qp_type = IBV_QPT_UC;
    init_attr.sq_sig_all = 0;

    qp = ibv_create_qp(pd, &init_attr);
    if (!qp) {
      std::cerr << "Failed to create queue pair: " << std::strerror(errno)
                << std::endl;
      break;
    }

    if (!modify_qp_to_init(qp, opts.ib_port)) {
      std::cerr << "Failed to move QP to INIT" << std::endl;
      break;
    }

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, opts.ib_port, &port_attr)) {
      std::cerr << "ibv_query_port failed" << std::endl;
      break;
    }

    ibv_gid gid{};
    if (ibv_query_gid(ctx, opts.ib_port, opts.gid_index, &gid)) {
      std::cerr << "ibv_query_gid failed" << std::endl;
      break;
    }

    std::random_device rd;
    uint32_t psn = rd() & 0xFFFFFF;

    WireQPInfo local{};
    local.lid = port_attr.lid;
    local.qp_num = qp->qp_num;
    local.psn = psn;
    std::memcpy(local.gid, gid.raw, 16);

    listen_fd = create_listen_socket(opts.listen_port);
    if (listen_fd < 0) {
      break;
    }
    std::cout << "Server waiting on control port " << opts.listen_port << std::endl;
    control_fd = accept_control(listen_fd);
    if (control_fd < 0) {
      break;
    }

    WireQPInfo remote{};
    if (!exchange_qp_info(control_fd, local, remote, true)) {
      std::cerr << "Failed to exchange QP information" << std::endl;
      break;
    }

    // Exchange test parameters - server uses client's values
    WireTestParams local_params{};
    local_params.iterations = static_cast<uint32_t>(opts.iterations);
    local_params.message_size = static_cast<uint32_t>(opts.message_size);
    WireTestParams remote_params{};
    if (!exchange_test_params(control_fd, local_params, remote_params, true)) {
      std::cerr << "Failed to exchange test parameters" << std::endl;
      break;
    }

    // Use client's parameters
    iterations = static_cast<int>(remote_params.iterations);
    msg_size = static_cast<std::size_t>(remote_params.message_size);
    std::cout << "Using client parameters: iterations=" << iterations
              << " message_size=" << msg_size << std::endl;

    const std::size_t buf_size = std::max(msg_size, static_cast<std::size_t>(4096));

    // Allocate multiple receive buffers for sliding window
    bool alloc_ok = true;
    for (int i = 0; i < kRecvWindowSize; ++i) {
      recv_bufs[i] = allocate_page_aligned(buf_size);
      if (!recv_bufs[i]) {
        std::cerr << "Failed to allocate page-aligned recv buffer " << i << std::endl;
        alloc_ok = false;
        break;
      }
      std::memset(recv_bufs[i], 0, buf_size);
    }
    if (!alloc_ok) break;

    send_buf = allocate_page_aligned(buf_size);
    if (!send_buf) {
      std::cerr << "Failed to allocate page-aligned send buffer" << std::endl;
      break;
    }
    std::memset(send_buf, 0, buf_size);

    int mr_access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    bool mr_ok = true;
    for (int i = 0; i < kRecvWindowSize; ++i) {
      recv_mrs[i] = ibv_reg_mr(pd, recv_bufs[i], buf_size, mr_access);
      if (!recv_mrs[i]) {
        std::cerr << "Failed to register recv memory " << i << std::endl;
        mr_ok = false;
        break;
      }
    }
    if (!mr_ok) break;

    send_mr = ibv_reg_mr(pd, send_buf, buf_size, mr_access);
    if (!send_mr) {
      std::cerr << "Failed to register send memory" << std::endl;
      break;
    }
    std::cout << "Registered " << kRecvWindowSize << " recv buffers + 1 send buffer, msg_size=" << msg_size << std::endl;

    ibv_gid remote_gid{};
    std::memcpy(remote_gid.raw, remote.gid, 16);
    bool use_gid = is_gid_nonzero(remote_gid);
    if (!modify_qp_to_rtr(qp, remote, opts.ib_port, use_gid, opts.gid_index)) {
      std::cerr << "Failed to move QP to RTR" << std::endl;
      break;
    }
    if (!modify_qp_to_rts(qp, psn)) {
      std::cerr << "Failed to move QP to RTS" << std::endl;
      break;
    }

    // Pre-post all receive buffers (sliding window)
    bool post_ok = true;
    for (int i = 0; i < kRecvWindowSize; ++i) {
      // Use wr_id to identify which buffer slot (1-based to avoid 0)
      if (!post_receive(qp, recv_bufs[i], recv_mrs[i], static_cast<uint64_t>(i + 1), msg_size)) {
        std::cerr << "Failed to post initial receive " << i << std::endl;
        post_ok = false;
        break;
      }
    }
    if (!post_ok) break;

    std::cout << "Server ready, QP in RTS state, pre-posted " << kRecvWindowSize
              << " receives, waiting for " << iterations << " messages..." << std::endl;

    // Synchronization barrier: let client know server is ready
    uint8_t ready = 1;
    if (!send_all(control_fd, &ready, 1) || !recv_all(control_fd, &ready, 1)) {
      std::cerr << "Failed to synchronize ready state" << std::endl;
      break;
    }
    std::cout << "Synchronized with client, ready to receive" << std::endl;

    int handled = 0;
    int outstanding_sends = 0;
    const int max_outstanding_sends = max_wr / 2;  // Leave room for receives
    bool success = true;

    while (handled < iterations) {
      ibv_wc wc{};
      if (!poll_completion(cq, wc)) {
        success = false;
        break;
      }

      // Handle receive completion
      if (wc.wr_id >= 1 && wc.wr_id <= static_cast<uint64_t>(kRecvWindowSize) &&
          (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
        int buf_idx = static_cast<int>(wc.wr_id) - 1;

        // Drain send completions if we're at the limit
        while (outstanding_sends >= max_outstanding_sends) {
          ibv_wc send_wc{};
          int num = ibv_poll_cq(cq, 1, &send_wc);
          if (num < 0) {
            std::cerr << "ibv_poll_cq failed while draining sends" << std::endl;
            success = false;
            break;
          }
          if (num > 0 && send_wc.wr_id >= 100 && send_wc.opcode == IBV_WC_SEND) {
            --outstanding_sends;
          }
        }
        if (!success) break;

        // Copy received data to send buffer and echo back
        std::memcpy(send_buf, recv_bufs[buf_idx], msg_size);
        if (!post_send(qp, send_buf, send_mr, 100 + handled, msg_size)) {
          std::cerr << "Failed to post echo (outstanding=" << outstanding_sends << ")" << std::endl;
          success = false;
          break;
        }
        ++outstanding_sends;
        ++handled;

        // Immediately re-post the receive buffer
        if (!post_receive(qp, recv_bufs[buf_idx], recv_mrs[buf_idx], wc.wr_id, msg_size)) {
          success = false;
          break;
        }
      }
      // Handle send completion
      else if (wc.wr_id >= 100 && wc.opcode == IBV_WC_SEND) {
        --outstanding_sends;
      }
    }

    // Drain any outstanding send completions
    while (outstanding_sends > 0 && success) {
      ibv_wc wc{};
      if (!poll_completion(cq, wc)) {
        success = false;
        break;
      }
      if (wc.wr_id >= 100 && wc.opcode == IBV_WC_SEND) {
        --outstanding_sends;
      }
    }

    if (success && handled == iterations) {
      std::cout << "Server completed " << handled << " iterations successfully" << std::endl;
      ret = 0;
    }
  } while (false);

  if (control_fd >= 0) {
    ::close(control_fd);
  }
  if (listen_fd >= 0) {
    ::close(listen_fd);
  }
  if (send_mr) {
    ibv_dereg_mr(send_mr);
  }
  for (int i = 0; i < kRecvWindowSize; ++i) {
    if (recv_mrs[i]) {
      ibv_dereg_mr(recv_mrs[i]);
    }
  }
  if (send_buf) {
    free(send_buf);
  }
  for (int i = 0; i < kRecvWindowSize; ++i) {
    if (recv_bufs[i]) {
      free(recv_bufs[i]);
    }
  }
  if (qp) {
    ibv_destroy_qp(qp);
  }
  if (cq) {
    ibv_destroy_cq(cq);
  }
  if (pd) {
    ibv_dealloc_pd(pd);
  }
  if (ctx) {
    ibv_close_device(ctx);
  }
  if (list) {
    ibv_free_device_list(list);
  }
  return ret;
}

int run_client(const Options& opts) {
  ibv_device** list = nullptr;
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_qp* qp = nullptr;
  void* recv_buf = nullptr;
  void* send_buf = nullptr;
  ibv_mr* recv_mr = nullptr;
  ibv_mr* send_mr = nullptr;
  int control_fd = -1;
  int ret = 1;
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

    // Increase CQ size to handle more completions
    int cq_entries = std::max(1, std::min<int>(dev_attr.max_cqe, 256));
    cq = ibv_create_cq(ctx, cq_entries, nullptr, nullptr, 0);
    if (!cq) {
      std::cerr << "Failed to create completion queue" << std::endl;
      break;
    }
    std::cout << "Created CQ with " << cq_entries << " entries" << std::endl;

    ibv_qp_init_attr init_attr{};
    init_attr.qp_context = ctx;
    init_attr.send_cq = cq;
    init_attr.recv_cq = cq;
    init_attr.srq = nullptr;
    int max_wr = std::max(1, std::min<int>(dev_attr.max_qp_wr, 64));
    int max_sge = std::max(1, std::min<int>(dev_attr.max_sge, 1));
    init_attr.cap.max_send_wr = max_wr;
    init_attr.cap.max_recv_wr = max_wr;
    init_attr.cap.max_send_sge = max_sge;
    init_attr.cap.max_recv_sge = max_sge;
    init_attr.cap.max_inline_data = 0;
    init_attr.qp_type = IBV_QPT_UC;
    init_attr.sq_sig_all = 0;

    qp = ibv_create_qp(pd, &init_attr);
    if (!qp) {
      std::cerr << "Failed to create queue pair: " << std::strerror(errno)
                << std::endl;
      break;
    }

    if (!modify_qp_to_init(qp, opts.ib_port)) {
      std::cerr << "Failed to move QP to INIT" << std::endl;
      break;
    }

    ibv_port_attr port_attr{};
    if (ibv_query_port(ctx, opts.ib_port, &port_attr)) {
      std::cerr << "ibv_query_port failed" << std::endl;
      break;
    }

    ibv_gid gid{};
    if (ibv_query_gid(ctx, opts.ib_port, opts.gid_index, &gid)) {
      std::cerr << "ibv_query_gid failed" << std::endl;
      break;
    }

    std::random_device rd;
    uint32_t psn = rd() & 0xFFFFFF;

    WireQPInfo local{};
    local.lid = port_attr.lid;
    local.qp_num = qp->qp_num;
    local.psn = psn;
    std::memcpy(local.gid, gid.raw, 16);

    control_fd = connect_control(opts.connect_host, opts.connect_port);
    if (control_fd < 0) {
      break;
    }

    WireQPInfo remote{};
    if (!exchange_qp_info(control_fd, local, remote, false)) {
      std::cerr << "Failed to exchange QP information" << std::endl;
      break;
    }

    // Exchange test parameters - client sends, server will use them
    WireTestParams local_params{};
    local_params.iterations = static_cast<uint32_t>(opts.iterations);
    local_params.message_size = static_cast<uint32_t>(opts.message_size);
    WireTestParams remote_params{};
    if (!exchange_test_params(control_fd, local_params, remote_params, false)) {
      std::cerr << "Failed to exchange test parameters" << std::endl;
      break;
    }
    std::cout << "Sent test parameters to server: iterations=" << opts.iterations
              << " message_size=" << msg_size << std::endl;

    recv_buf = allocate_page_aligned(buf_size);
    send_buf = allocate_page_aligned(buf_size);
    if (!recv_buf || !send_buf) {
      std::cerr << "Failed to allocate page-aligned buffers" << std::endl;
      break;
    }
    std::memset(recv_buf, 0, buf_size);
    std::memset(send_buf, 0, buf_size);

    int mr_access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    recv_mr = ibv_reg_mr(pd, recv_buf, buf_size, mr_access);
    send_mr = ibv_reg_mr(pd, send_buf, buf_size, mr_access);
    if (!recv_mr || !send_mr) {
      std::cerr << "Failed to register memory" << std::endl;
      break;
    }
    std::cout << "Registered buffers: recv lkey=" << recv_mr->lkey
              << " send lkey=" << send_mr->lkey
              << " msg_size=" << msg_size << std::endl;

    ibv_gid remote_gid{};
    std::memcpy(remote_gid.raw, remote.gid, 16);
    bool use_gid = is_gid_nonzero(remote_gid);
    if (!modify_qp_to_rtr(qp, remote, opts.ib_port, use_gid, opts.gid_index)) {
      std::cerr << "Failed to move QP to RTR" << std::endl;
      break;
    }
    if (!modify_qp_to_rts(qp, psn)) {
      std::cerr << "Failed to move QP to RTS" << std::endl;
      break;
    }

    if (!post_receive(qp, recv_buf, recv_mr, 1, msg_size)) {
      std::cerr << "Failed to post initial receive" << std::endl;
      break;
    }
    std::cout << "Client ready, QP in RTS state, starting to send..." << std::endl;

    // Synchronization barrier: wait for server to be ready
    uint8_t ready = 1;
    if (!send_all(control_fd, &ready, 1) || !recv_all(control_fd, &ready, 1)) {
      std::cerr << "Failed to synchronize ready state" << std::endl;
      break;
    }
    std::cout << "Synchronized with server, starting to send" << std::endl;

    fill_payload(send_buf, msg_size);
    MessageHeader* send_hdr = static_cast<MessageHeader*>(send_buf);
    MessageHeader* recv_hdr = static_cast<MessageHeader*>(recv_buf);
    uint64_t sequence = 0;
    bool success = true;
    std::vector<double> latencies;
    constexpr int warmup_iters = 100;

    // Packet loss tracking
    uint64_t expected_recv_seq = 1;  // Next expected sequence in received echo
    uint64_t lost_packets = 0;       // Packets sent but no echo received (gaps in recv seq)
    uint64_t out_of_order = 0;       // Packets received out of order
    uint64_t duplicate_packets = 0;  // Duplicate sequence numbers received
    std::vector<uint64_t> received_seqs;  // Track all received sequences for analysis

    // Bandwidth measurement
    uint64_t test_start_ns = wall_time_ns();

    if (opts.flood_outstanding > 0) {
      // Flood mode: send with limited outstanding to avoid saturating path
      const int max_outstanding = opts.flood_outstanding;
      std::cout << "FLOOD MODE: sending without waiting for echoes (max_outstanding="
                << max_outstanding << ")" << std::endl;
      int outstanding_sends = 0;
      int progress_interval = opts.iterations / 10;
      if (progress_interval < 1) progress_interval = 1;

      for (int iter = 0; iter < opts.iterations; ++iter) {
        // Progress indicator
        if (iter > 0 && iter % progress_interval == 0) {
          std::cout << "  Sent " << iter << "/" << opts.iterations << " messages" << std::endl;
        }

        // Poll completions to free up send queue (non-blocking)
        int poll_attempts = 0;
        while (outstanding_sends >= max_outstanding) {
          ibv_wc wc{};
          int num = ibv_poll_cq(cq, 1, &wc);
          if (num < 0) {
            std::cerr << "ibv_poll_cq failed" << std::endl;
            success = false;
            break;
          }
          if (num > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
              std::cerr << "Completion error: status=" << wc.status
                        << " wr_id=" << wc.wr_id << " opcode=" << wc.opcode << std::endl;
              success = false;
              break;
            }
            if (wc.opcode == IBV_WC_SEND) {
              --outstanding_sends;
            } else if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              // Got a recv completion - track RTT and stats
              uint64_t now_ns = wall_time_ns();
              uint64_t recv_seq = recv_hdr->sequence;
              double latency_us = static_cast<double>(now_ns - recv_hdr->send_time_ns) / 1000.0;

              received_seqs.push_back(recv_seq);
              if (received_seqs.size() > static_cast<size_t>(warmup_iters)) {
                latencies.push_back(latency_us);
              }

              // Repost receive buffer
              if (!post_receive(qp, recv_buf, recv_mr, 1, msg_size)) {
                std::cerr << "Failed to repost receive in flood mode" << std::endl;
              }
            }
          }
          ++poll_attempts;
          if (poll_attempts > 10000000) {
            std::cerr << "TIMEOUT waiting for send completion, outstanding="
                      << outstanding_sends << std::endl;
            success = false;
            break;
          }
        }
        if (!success) break;

        ++sequence;
        send_hdr->sequence = sequence;
        send_hdr->send_time_ns = wall_time_ns();

        if (!post_send(qp, send_buf, send_mr, 2, msg_size)) {
          std::cerr << "Failed to post send at iter " << iter << std::endl;
          success = false;
          break;
        }
        ++outstanding_sends;
      }

      std::cout << "Draining " << outstanding_sends << " outstanding sends..." << std::endl;
      // Drain remaining send completions
      while (outstanding_sends > 0 && success) {
        ibv_wc wc{};
        int num = ibv_poll_cq(cq, 1, &wc);
        if (num < 0) {
          std::cerr << "ibv_poll_cq failed during drain" << std::endl;
          success = false;
          break;
        }
        if (num > 0) {
          if (wc.opcode == IBV_WC_SEND) {
            --outstanding_sends;
          } else if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
            // Track recv during drain too
            uint64_t now_ns = wall_time_ns();
            double latency_us = static_cast<double>(now_ns - recv_hdr->send_time_ns) / 1000.0;
            received_seqs.push_back(recv_hdr->sequence);
            if (received_seqs.size() > static_cast<size_t>(warmup_iters)) {
              latencies.push_back(latency_us);
            }
            if (!post_receive(qp, recv_buf, recv_mr, 1, msg_size)) {
              std::cerr << "Failed to repost receive during drain" << std::endl;
            }
          }
        }
      }

      // Wait for remaining echoes from server (with timeout)
      std::cout << "Waiting for remaining echoes (received " << received_seqs.size()
                << "/" << sequence << ")..." << std::endl;
      int drain_timeout = 0;
      while (received_seqs.size() < sequence && drain_timeout < 10000000 && success) {
        ibv_wc wc{};
        int num = ibv_poll_cq(cq, 1, &wc);
        if (num < 0) {
          success = false;
          break;
        }
        if (num > 0 && (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
          uint64_t now_ns = wall_time_ns();
          double latency_us = static_cast<double>(now_ns - recv_hdr->send_time_ns) / 1000.0;
          received_seqs.push_back(recv_hdr->sequence);
          if (received_seqs.size() > static_cast<size_t>(warmup_iters)) {
            latencies.push_back(latency_us);
          }
          if (!post_receive(qp, recv_buf, recv_mr, 1, msg_size)) {
            std::cerr << "Failed to repost receive" << std::endl;
          }
          drain_timeout = 0;  // Reset timeout on progress
        } else {
          ++drain_timeout;
        }
      }
      std::cout << "Flood complete: sent " << sequence << ", received "
                << received_seqs.size() << " echoes" << std::endl;

    } else {
      // Normal ping-pong mode: wait for echo before sending next
      for (int iter = 0; iter < opts.iterations; ++iter) {
        ++sequence;
        send_hdr->sequence = sequence;
        send_hdr->send_time_ns = wall_time_ns();

        if (!post_send(qp, send_buf, send_mr, 2, msg_size)) {
          std::cerr << "Failed to post send" << std::endl;
          success = false;
          break;
        }

        bool have_send = false;
        bool have_recv = false;
        while (!have_send || !have_recv) {
          ibv_wc wc{};
          if (!poll_completion(cq, wc)) {
            success = false;
            have_send = have_recv = false;
            break;
          }
          if (wc.wr_id == 2 && wc.opcode == IBV_WC_SEND) {
            have_send = true;
          } else if (wc.wr_id == 1 &&
                     (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
            have_recv = true;
          }
        }
        if (!have_send || !have_recv) {
          success = false;
          break;
        }

        uint64_t now_ns = wall_time_ns();
        double latency_us = static_cast<double>(now_ns - recv_hdr->send_time_ns) / 1000.0;

        // Check sequence number for packet loss detection
        uint64_t recv_seq = recv_hdr->sequence;
        received_seqs.push_back(recv_seq);

        if (recv_seq == expected_recv_seq) {
          // Normal case: received expected sequence
          ++expected_recv_seq;
        } else if (recv_seq > expected_recv_seq) {
          // Gap detected: some packets were lost
          uint64_t gap = recv_seq - expected_recv_seq;
          lost_packets += gap;
          std::cerr << "Packet loss detected: expected seq " << expected_recv_seq
                    << ", got " << recv_seq << " (lost " << gap << " packets)" << std::endl;
          expected_recv_seq = recv_seq + 1;
        } else {
          // recv_seq < expected_recv_seq: out of order or duplicate
          // Check if we've seen this sequence before
          bool is_duplicate = false;
          for (size_t i = 0; i + 1 < received_seqs.size(); ++i) {
            if (received_seqs[i] == recv_seq) {
              is_duplicate = true;
              break;
            }
          }
          if (is_duplicate) {
            ++duplicate_packets;
            std::cerr << "Duplicate packet: seq " << recv_seq << std::endl;
          } else {
            ++out_of_order;
            std::cerr << "Out-of-order packet: expected seq " << expected_recv_seq
                      << ", got " << recv_seq << std::endl;
          }
        }

        if (iter >= warmup_iters) {
          latencies.push_back(latency_us);
        }

        if (!post_receive(qp, recv_buf, recv_mr, 1, msg_size)) {
          success = false;
          break;
        }
      }
    }

    if (success && sequence == static_cast<uint64_t>(opts.iterations)) {
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

      // Packet loss statistics
      std::cout << "\n=== Packet Loss Statistics ===" << std::endl;
      std::cout << "Total sent:       " << sequence << std::endl;
      std::cout << "Total received:   " << received_seqs.size() << std::endl;
      std::cout << "Lost packets:     " << lost_packets << std::endl;
      std::cout << "Out-of-order:     " << out_of_order << std::endl;
      std::cout << "Duplicates:       " << duplicate_packets << std::endl;
      if (sequence > 0) {
        double loss_rate = 100.0 * static_cast<double>(lost_packets) / static_cast<double>(sequence);
        std::cout << "Loss rate:        " << std::fixed << std::setprecision(4) << loss_rate << " %" << std::endl;
      }

      // Bandwidth statistics
      uint64_t test_end_ns = wall_time_ns();
      double test_duration_s = static_cast<double>(test_end_ns - test_start_ns) / 1e9;
      // Total bytes: each iteration sends 1 message and receives 1 echo (2 x msg_size)
      uint64_t total_bytes = sequence * 2 * msg_size;
      double bandwidth_gbps = static_cast<double>(total_bytes) / test_duration_s / 1e9;
      double msg_rate = static_cast<double>(sequence) / test_duration_s;

      std::cout << "\n=== Bandwidth Statistics ===" << std::endl;
      std::cout << "Test duration:    " << std::fixed << std::setprecision(3) << test_duration_s << " s" << std::endl;
      std::cout << "Message size:     " << msg_size << " bytes" << std::endl;
      std::cout << "Total data:       " << std::fixed << std::setprecision(2)
                << static_cast<double>(total_bytes) / 1e9 << " GB (send+recv)" << std::endl;
      std::cout << "Bandwidth:        " << std::fixed << std::setprecision(4) << bandwidth_gbps << " GB/s" << std::endl;
      std::cout << "Message rate:     " << std::fixed << std::setprecision(0) << msg_rate << " msg/s" << std::endl;
    }
  } while (false);

  if (control_fd >= 0) {
    ::close(control_fd);
  }
  if (send_mr) {
    ibv_dereg_mr(send_mr);
  }
  if (recv_mr) {
    ibv_dereg_mr(recv_mr);
  }
  if (send_buf) {
    free(send_buf);
  }
  if (recv_buf) {
    free(recv_buf);
  }
  if (qp) {
    ibv_destroy_qp(qp);
  }
  if (cq) {
    ibv_destroy_cq(cq);
  }
  if (pd) {
    ibv_dealloc_pd(pd);
  }
  if (ctx) {
    ibv_close_device(ctx);
  }
  if (list) {
    ibv_free_device_list(list);
  }
  return ret;
}

void usage(const char* prog) {
  std::cerr << "Usage:" << std::endl;
  std::cerr << "  " << prog
            << " --server --listen <port> [--device <name>] [--ib-port <n>]"
            << " [--gid-index <n>]" << std::endl;
  std::cerr << "  " << prog
            << " --client --connect <host:port> [--device <name>] [--ib-port <n>]"
            << " [--gid-index <n>] [--iterations <n>] [--message-size <bytes>] [--flood <n>]" << std::endl;
  std::cerr << "  Default message size: " << kDefaultMessageSize << " bytes, max: "
            << kMaxMessageSize << " bytes" << std::endl;
  std::cerr << "  --flood <n>: Flood mode with n packets in flight (default 2 if no value)" << std::endl;
  std::cerr << "  Note: Server uses client's iteration count and message size" << std::endl;
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
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        opts.flood_outstanding = std::stoi(argv[++i]);
        if (opts.flood_outstanding < 1) opts.flood_outstanding = 2;
      } else {
        opts.flood_outstanding = 2;  // Default to 2 if no value given
      }
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
