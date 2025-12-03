// ANWMLL testing Tensor Parallel workflows for RDMA driver

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace mx = mlx::core;
constexpr const char* kBenchmarkVersion = "0.0.13";

struct Args {
  int hidden_size = 4096;
  int intermediate_size = 14336;
  int batch_size = 1;
  int iterations = 1000;
  int warmup = 50;
  int num_blocks = 8;
  std::string dtype = "bfloat16";
  std::optional<std::string> metal_trace;
  bool single_host = false;
  bool skip_single = false;
  int single_host_runner = 0;
  bool emit_json = false;
  std::optional<std::string> rank_env;
  std::optional<std::string> coordinator_env;
  std::optional<std::string> devices_env;
  std::optional<std::string> verbose_env;
  bool profile_comms = false;
  bool comm_only = false;
  bool cpu_only = false;
  bool fake_compute = false;
  bool send_recv = false;
  bool custom_allreduce = false;
  bool hybrid = false;  // GPU compute + CPU AllReduce
  bool sync_bench = false;  // Measure pure eval/sync overhead
};

struct Stats {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
};

struct CommProfile {
  Stats total;
  Stats compute;
  Stats comm;
  int num_allreduce_calls = 0;
  size_t bytes_per_allreduce = 0;
};

struct BlockWeights {
  mx::array w_up;
  mx::array w_down;
};

mx::Dtype parse_dtype(const std::string& name) {
  if (name == "float32") {
    return mx::float32;
  } else if (name == "bfloat16") {
    return mx::bfloat16;
  } else if (name == "float16") {
    return mx::float16;
  }
  throw std::runtime_error("Unsupported dtype: " + name);
}

Args parse_args(int argc, char** argv) {
  Args args;
  auto consume_value = [&](int& index) -> std::string {
    if (index + 1 >= argc) {
      throw std::runtime_error("Missing value for argument " +
                               std::string(argv[index]));
    }
    return std::string(argv[++index]);
  };

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--hidden-size") {
      args.hidden_size = std::stoi(consume_value(i));
    } else if (arg == "--intermediate-size") {
      args.intermediate_size = std::stoi(consume_value(i));
    } else if (arg == "--batch-size") {
      args.batch_size = std::stoi(consume_value(i));
    } else if (arg == "--iterations") {
      args.iterations = std::stoi(consume_value(i));
    } else if (arg == "--warmup") {
      args.warmup = std::stoi(consume_value(i));
    } else if (arg == "--num-blocks") {
      args.num_blocks = std::stoi(consume_value(i));
    } else if (arg == "--dtype") {
      args.dtype = consume_value(i);
    } else if (arg == "--metal-trace") {
      args.metal_trace = consume_value(i);
    } else if (arg == "--single-host") {
      args.single_host = true;
    } else if (arg == "--skip-single") {
      args.skip_single = true;
    } else if (arg == "--single-host-runner") {
      args.single_host_runner = std::stoi(consume_value(i));
    } else if (arg == "--emit-json") {
      args.emit_json = true;
    } else if (arg == "--rank") {
      args.rank_env = consume_value(i);
    } else if (arg == "--coordinator") {
      args.coordinator_env = consume_value(i);
    } else if (arg == "--devices") {
      args.devices_env = consume_value(i);
    } else if (arg == "--verbose") {
      args.verbose_env = consume_value(i);
    } else if (arg == "--profile-comms") {
      args.profile_comms = true;
    } else if (arg == "--comm-only") {
      args.comm_only = true;
    } else if (arg == "--cpu") {
      args.cpu_only = true;
    } else if (arg == "--fake-compute") {
      args.fake_compute = true;
    } else if (arg == "--send-recv") {
      args.send_recv = true;
    } else if (arg == "--custom-allreduce") {
      args.custom_allreduce = true;
    } else if (arg == "--hybrid") {
      args.hybrid = true;
    } else if (arg == "--sync-bench") {
      args.sync_bench = true;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return args;
}

mx::array gelu(const mx::array& x) {
  float coeff = 0.7978845608f;
  float alpha = 0.044715f;
  auto dtype = x.dtype();
  auto coeff_arr = mx::array(coeff, dtype);
  auto alpha_arr = mx::array(alpha, dtype);
  mx::array half = mx::array(0.5f, dtype);
  mx::array one = mx::array(1.0f, dtype);

  auto x_sq = mx::multiply(x, x);
  auto x_cu = mx::multiply(x_sq, x);
  auto inner = coeff_arr * (x + alpha_arr * x_cu);
  auto tanh_inner = mx::tanh(inner);
  return half * mx::multiply(x, (one + tanh_inner));
}

mx::array apply_blocks(const mx::array& input,
                       const std::vector<BlockWeights>& blocks,
                       const mx::distributed::Group& group) {
  mx::array current = input;
  for (const auto& block : blocks) {
    auto up = mx::matmul(current, mx::transpose(block.w_up));
    auto activated = gelu(up);
    auto partial = mx::matmul(activated, mx::transpose(block.w_down));
    if (group.size() > 1) {
      current = mx::distributed::all_sum(partial, group);
    } else {
      current = partial;
    }
  }
  return current;
}

mx::array apply_blocks_local(const mx::array& input,
                             const std::vector<BlockWeights>& blocks) {
  mx::array current = input;
  for (const auto& block : blocks) {
    auto up = mx::matmul(current, mx::transpose(block.w_up));
    auto activated = gelu(up);
    current = mx::matmul(activated, mx::transpose(block.w_down));
  }
  return current;
}

// Hybrid version: GPU compute + explicit sync before AllReduce
// On Apple Silicon unified memory, we explicitly sync GPU before AllReduce
// to measure overhead of per-block synchronization
mx::array apply_blocks_hybrid(const mx::array& input,
                              const std::vector<BlockWeights>& blocks,
                              const mx::distributed::Group& group) {
  mx::array current = input;

  for (const auto& block : blocks) {
    // GPU compute (on default GPU stream)
    auto up = mx::matmul(current, mx::transpose(block.w_up));
    auto activated = gelu(up);
    auto partial = mx::matmul(activated, mx::transpose(block.w_down));

    // Explicit sync GPU before AllReduce - this is what causes the ~200us overhead
    mx::eval(partial);

    // AllReduce (JACCL handles CPU communication internally)
    if (group.size() > 1) {
      current = mx::distributed::all_sum(partial, group);
      mx::eval(current);  // Ensure AllReduce completes before next iteration
    } else {
      current = partial;
    }
  }
  return current;
}

// Fake compute version: skips actual matmul/activation, sends zeros for AllReduce
// Used to measure pure communication overhead without GPU compute interference
mx::array apply_blocks_fake(const mx::array& input,
                            const std::vector<BlockWeights>& blocks,
                            const mx::distributed::Group& group) {
  mx::array current = input;
  for (size_t i = 0; i < blocks.size(); ++i) {
    // Create zeros with same shape as output would have (batch x hidden)
    auto fake_partial = mx::zeros_like(current);
    mx::eval(fake_partial);  // Ensure tensor is materialized before AllReduce

    if (group.size() > 1) {
      current = mx::distributed::all_sum(fake_partial, group);
    } else {
      current = fake_partial;
    }
  }
  return current;
}

// Profiled version: returns (output, compute_time_us, comm_time_us)
std::tuple<mx::array, double, double> apply_blocks_profiled(
    const mx::array& input,
    const std::vector<BlockWeights>& blocks,
    const mx::distributed::Group& group) {
  mx::array current = input;
  double compute_us = 0.0;
  double comm_us = 0.0;

  for (const auto& block : blocks) {
    // Time compute: up projection + activation + down projection
    auto compute_start = std::chrono::high_resolution_clock::now();
    auto up = mx::matmul(current, mx::transpose(block.w_up));
    auto activated = gelu(up);
    auto partial = mx::matmul(activated, mx::transpose(block.w_down));
    mx::eval(partial);
    auto compute_end = std::chrono::high_resolution_clock::now();
    compute_us +=
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            compute_end - compute_start)
            .count();

    // Time communication: all_sum
    if (group.size() > 1) {
      auto comm_start = std::chrono::high_resolution_clock::now();
      current = mx::distributed::all_sum(partial, group);
      mx::eval(current);
      auto comm_end = std::chrono::high_resolution_clock::now();
      comm_us +=
          std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
              comm_end - comm_start)
              .count();
    } else {
      current = partial;
    }
  }
  return {current, compute_us, comm_us};
}

std::vector<BlockWeights> gather_full_blocks(
    const std::vector<BlockWeights>& blocks,
    const mx::distributed::Group& group,
    bool keep_data) {
  std::vector<BlockWeights> full;
  if (keep_data) {
    full.reserve(blocks.size());
  }
  for (const auto& block : blocks) {
    if (keep_data) {
      std::vector<mx::array> ups;
      ups.reserve(group.size());
      ups.push_back(block.w_up);
      for (int src = 1; src < group.size(); ++src) {
        auto recv = mx::distributed::recv_like(block.w_up, src, group);
        ups.push_back(recv);
      }
      auto up_full = mx::concatenate(ups, 0);
      mx::eval(up_full);

      std::vector<mx::array> downs;
      downs.reserve(group.size());
      downs.push_back(block.w_down);
      for (int src = 1; src < group.size(); ++src) {
        auto recv = mx::distributed::recv_like(block.w_down, src, group);
        downs.push_back(recv);
      }
      auto down_full = mx::concatenate(downs, 1);
      mx::eval(down_full);

      full.push_back(BlockWeights{up_full, down_full});
    } else {
      auto token_up = mx::distributed::send(block.w_up, 0, group);
      auto token_down = mx::distributed::send(block.w_down, 0, group);
      mx::eval(token_up, token_down);
    }
  }
  auto barrier =
      mx::distributed::all_sum(mx::zeros({1}, mx::float32), group);
  mx::eval(barrier);
  return full;
}

Stats summarize(const std::vector<double>& times) {
  Stats stats;
  stats.min = *std::min_element(times.begin(), times.end());
  stats.max = *std::max_element(times.begin(), times.end());
  stats.mean =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  auto sorted = times;
  std::sort(sorted.begin(), sorted.end());
  auto pick = [&](double frac) {
    size_t idx = static_cast<size_t>(frac * static_cast<double>(sorted.size()));
    idx = std::min(idx, sorted.size() - 1);
    return sorted[idx];
  };
  stats.p50 = pick(0.5);
  stats.p95 = pick(0.95);
  stats.p99 = pick(0.99);
  return stats;
}

std::vector<BlockWeights> create_blocks(const Args& args,
                                        const mx::distributed::Group& group,
                                        mx::Dtype dtype) {
  if (args.intermediate_size % group.size() != 0) {
    throw std::runtime_error(
        "intermediate_size must be divisible by world size");
  }
  int intermediate_per_rank = args.intermediate_size / group.size();
  std::vector<BlockWeights> blocks;
  blocks.reserve(args.num_blocks);
  for (int i = 0; i < args.num_blocks; ++i) {
    BlockWeights block{
        mx::random::normal({intermediate_per_rank, args.hidden_size}, dtype),
        mx::random::normal({args.hidden_size, intermediate_per_rank}, dtype)};
    mx::eval(block.w_up, block.w_down);
    blocks.push_back(std::move(block));
  }
  return blocks;
}

Stats benchmark(const std::vector<BlockWeights>& blocks,
                const mx::array& input,
                const mx::distributed::Group& group,
                const Args& args,
                bool verbose) {
  if (verbose) {
    std::cout << "[verbose] Warmup iterations: " << args.warmup << std::endl;
  }
  for (int i = 0; i < args.warmup; ++i) {
    auto out = apply_blocks(input, blocks, group);
    mx::eval(out);
  }

  bool capture_started = false;
  if (args.metal_trace.has_value()) {
    if (std::filesystem::exists(*args.metal_trace)) {
      throw std::runtime_error("Trace already exists: " + *args.metal_trace);
    }
    if (!std::getenv("MTL_CAPTURE_ENABLED")) {
      std::cerr << "WARNING: MTL_CAPTURE_ENABLED not set; capture may be empty"
                << std::endl;
    }
    mx::metal::start_capture(*args.metal_trace);
    capture_started = true;
    std::cout << "Started Metal capture: " << *args.metal_trace << std::endl;
  }

  std::vector<double> times;
  times.reserve(args.iterations);
  if (verbose) {
    std::cout << "[verbose] Timed iterations: " << args.iterations
              << std::endl;
  }
  for (int i = 0; i < args.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto out = apply_blocks(input, blocks, group);
    mx::eval(out);
    auto end = std::chrono::high_resolution_clock::now();
    double usec =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            end - start)
            .count();
    times.push_back(usec);
  }

  if (capture_started) {
    mx::metal::stop_capture();
    std::cout << "Metal capture saved to " << *args.metal_trace << std::endl;
  }

  return summarize(times);
}

// Benchmark hybrid mode: GPU compute + CPU AllReduce
Stats benchmark_hybrid(const std::vector<BlockWeights>& blocks,
                       const mx::array& input,
                       const mx::distributed::Group& group,
                       const Args& args,
                       bool verbose) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = args.batch_size * args.hidden_size * elem_size;

  if (verbose) {
    std::cout << "[hybrid] GPU compute + CPU AllReduce\n";
    std::cout << "[hybrid] Tensor shape: [" << args.batch_size << ", "
              << args.hidden_size << "]\n";
    std::cout << "[hybrid] Tensor size: " << tensor_bytes << " bytes ("
              << tensor_bytes / 1024.0 << " KB)\n";
    std::cout << "[hybrid] Num blocks: " << args.num_blocks << "\n";
    std::cout << "[hybrid] Warmup iterations: " << args.warmup << std::endl;
  }

  // Warmup
  for (int i = 0; i < args.warmup; ++i) {
    auto out = apply_blocks_hybrid(input, blocks, group);
    mx::eval(out);
  }

  std::vector<double> times;
  times.reserve(args.iterations);

  if (verbose) {
    std::cout << "[hybrid] Timed iterations: " << args.iterations << std::endl;
  }

  for (int i = 0; i < args.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto out = apply_blocks_hybrid(input, blocks, group);
    mx::eval(out);
    auto end = std::chrono::high_resolution_clock::now();
    double usec =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            end - start)
            .count();
    times.push_back(usec);
  }

  return summarize(times);
}

void print_hybrid_stats(const Stats& stats,
                        const mx::distributed::Group& group,
                        const Args& args) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = args.batch_size * args.hidden_size * elem_size;

  std::cout << "\n" << std::string(70, '=') << "\n"
            << "Hybrid Benchmark: GPU Compute + CPU AllReduce (rank " << group.rank() << "/"
            << group.size() << ")\n"
            << std::string(70, '=') << std::endl;

  std::cout << "\n*** GPU matmul/activation + CPU AllReduce (unified memory) ***\n\n";

  std::cout << "Configuration:\n"
            << "  Hidden size:       " << args.hidden_size << "\n"
            << "  Intermediate size: " << args.intermediate_size << "\n"
            << "  Batch size:        " << args.batch_size << "\n"
            << "  Tensor bytes:      " << tensor_bytes << " ("
            << tensor_bytes / 1024.0 << " KB)\n"
            << "  Num blocks:        " << args.num_blocks << "\n"
            << "  World size:        " << group.size() << "\n"
            << "  Iterations:        " << args.iterations << "\n\n";

  std::cout << "Total Iteration Latency:\n"
            << "  Mean:  " << stats.mean << " μs (" << stats.mean / 1000.0 << " ms)\n"
            << "  Min:   " << stats.min << " μs\n"
            << "  p50:   " << stats.p50 << " μs\n"
            << "  p95:   " << stats.p95 << " μs\n"
            << "  p99:   " << stats.p99 << " μs\n"
            << "  Max:   " << stats.max << " μs\n\n";

  double per_block = stats.mean / args.num_blocks;
  std::cout << "Per-Block Latency:\n"
            << "  Mean:  " << per_block << " μs\n";
}

// Sync benchmark: measures pure mx::eval() overhead (waitUntilCompleted latency)
void benchmark_sync(const Args& args, bool verbose) {
  std::cout << "\n" << std::string(70, '=') << "\n"
            << "Sync Benchmark: Pure mx::eval() Overhead\n"
            << std::string(70, '=') << std::endl;

  auto dtype = (args.dtype == "float32") ? mx::float32 : mx::bfloat16;

  // Test 1: Empty eval (no-op)
  std::cout << "\n1. Empty eval (already-evaluated array):\n";
  {
    auto arr = mx::zeros({1}, dtype);
    mx::eval(arr);  // Pre-evaluate

    std::vector<double> times;
    for (int i = 0; i < args.iterations; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      mx::eval(arr);  // Should be nearly instant - already evaluated
      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    auto stats = summarize(times);
    std::cout << "   Mean: " << stats.mean << " μs, Min: " << stats.min
              << " μs, p50: " << stats.p50 << " μs\n";
  }

  // Test 2: Minimal GPU op (zeros)
  std::cout << "\n2. Minimal GPU op (create zeros + eval):\n";
  {
    // Warmup
    for (int i = 0; i < args.warmup; ++i) {
      auto arr = mx::zeros({1}, dtype);
      mx::eval(arr);
    }

    std::vector<double> times;
    for (int i = 0; i < args.iterations; ++i) {
      auto arr = mx::zeros({1}, dtype);  // Lazy - not yet on GPU
      auto start = std::chrono::high_resolution_clock::now();
      mx::eval(arr);  // This triggers GPU work + waitUntilCompleted
      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    auto stats = summarize(times);
    std::cout << "   Mean: " << stats.mean << " μs, Min: " << stats.min
              << " μs, p50: " << stats.p50 << " μs\n";
  }

  // Test 3: Small matmul + eval
  std::cout << "\n3. Small matmul [64x64] @ [64x64] + eval:\n";
  {
    auto a = mx::random::normal({64, 64}, dtype);
    auto b = mx::random::normal({64, 64}, dtype);
    mx::eval(a, b);

    // Warmup
    for (int i = 0; i < args.warmup; ++i) {
      auto c = mx::matmul(a, b);
      mx::eval(c);
    }

    std::vector<double> times;
    for (int i = 0; i < args.iterations; ++i) {
      auto c = mx::matmul(a, b);  // Lazy
      auto start = std::chrono::high_resolution_clock::now();
      mx::eval(c);
      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    auto stats = summarize(times);
    std::cout << "   Mean: " << stats.mean << " μs, Min: " << stats.min
              << " μs, p50: " << stats.p50 << " μs\n";
  }

  // Test 4: Typical tensor size matmul
  std::cout << "\n4. Typical matmul [1x" << args.hidden_size << "] @ ["
            << args.hidden_size << "x" << args.intermediate_size / 2 << "] + eval:\n";
  {
    auto a = mx::random::normal({args.batch_size, args.hidden_size}, dtype);
    auto b = mx::random::normal({args.hidden_size, args.intermediate_size / 2}, dtype);
    mx::eval(a, b);

    // Warmup
    for (int i = 0; i < args.warmup; ++i) {
      auto c = mx::matmul(a, b);
      mx::eval(c);
    }

    std::vector<double> times;
    for (int i = 0; i < args.iterations; ++i) {
      auto c = mx::matmul(a, b);
      auto start = std::chrono::high_resolution_clock::now();
      mx::eval(c);
      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    auto stats = summarize(times);
    std::cout << "   Mean: " << stats.mean << " μs, Min: " << stats.min
              << " μs, p50: " << stats.p50 << " μs\n";
  }

  // Test 5: Multiple evals in sequence (simulates hybrid mode)
  std::cout << "\n5. " << args.num_blocks << " sequential evals (simulates hybrid mode):\n";
  {
    auto a = mx::random::normal({args.batch_size, args.hidden_size}, dtype);
    auto b = mx::random::normal({args.hidden_size, args.intermediate_size / 2}, dtype);
    mx::eval(a, b);

    // Warmup
    for (int w = 0; w < args.warmup; ++w) {
      for (int i = 0; i < args.num_blocks; ++i) {
        auto c = mx::matmul(a, b);
        mx::eval(c);
      }
    }

    std::vector<double> times;
    for (int iter = 0; iter < args.iterations; ++iter) {
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < args.num_blocks; ++i) {
        auto c = mx::matmul(a, b);
        mx::eval(c);
      }
      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    auto stats = summarize(times);
    double per_eval = stats.mean / args.num_blocks;
    std::cout << "   Total: " << stats.mean << " μs, Per-eval: " << per_eval << " μs\n";
  }

  // Test 6: Single eval of full graph (simulates pipelined mode)
  std::cout << "\n6. Single eval of " << args.num_blocks << " chained matmuls (pipelined):\n";
  {
    auto a = mx::random::normal({args.batch_size, args.hidden_size}, dtype);
    std::vector<mx::array> weights;
    for (int i = 0; i < args.num_blocks; ++i) {
      weights.push_back(mx::random::normal({args.hidden_size, args.hidden_size}, dtype));
    }
    mx::eval(a);
    for (auto& w : weights) mx::eval(w);

    // Warmup
    for (int w = 0; w < args.warmup; ++w) {
      auto current = a;
      for (int i = 0; i < args.num_blocks; ++i) {
        current = mx::matmul(current, weights[i]);
      }
      mx::eval(current);
    }

    std::vector<double> times;
    for (int iter = 0; iter < args.iterations; ++iter) {
      auto current = a;
      for (int i = 0; i < args.num_blocks; ++i) {
        current = mx::matmul(current, weights[i]);
      }
      auto start = std::chrono::high_resolution_clock::now();
      mx::eval(current);
      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }
    auto stats = summarize(times);
    std::cout << "   Total: " << stats.mean << " μs (single eval for all " << args.num_blocks << " ops)\n";
  }

  std::cout << "\n" << std::string(70, '-') << "\n"
            << "Conclusion: Compare Test 5 vs Test 6 to see eval overhead.\n"
            << "The difference is the cost of " << (args.num_blocks - 1)
            << " extra waitUntilCompleted() calls.\n"
            << std::string(70, '=') << "\n";
}

CommProfile benchmark_comms(const std::vector<BlockWeights>& blocks,
                            const mx::array& input,
                            const mx::distributed::Group& group,
                            const Args& args,
                            bool verbose) {
  CommProfile profile;
  profile.num_allreduce_calls = args.num_blocks;
  // bytes = batch_size * hidden_size * sizeof(dtype)
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  profile.bytes_per_allreduce = args.batch_size * args.hidden_size * elem_size;

  if (verbose) {
    std::cout << "[profile-comms] Warmup iterations: " << args.warmup
              << std::endl;
  }
  for (int i = 0; i < args.warmup; ++i) {
    auto [out, compute_us, comm_us] =
        apply_blocks_profiled(input, blocks, group);
    (void)compute_us;
    (void)comm_us;
  }

  std::vector<double> total_times;
  std::vector<double> compute_times;
  std::vector<double> comm_times;
  total_times.reserve(args.iterations);
  compute_times.reserve(args.iterations);
  comm_times.reserve(args.iterations);

  if (verbose) {
    std::cout << "[profile-comms] Timed iterations: " << args.iterations
              << std::endl;
  }
  for (int i = 0; i < args.iterations; ++i) {
    auto iter_start = std::chrono::high_resolution_clock::now();
    auto [out, compute_us, comm_us] =
        apply_blocks_profiled(input, blocks, group);
    auto iter_end = std::chrono::high_resolution_clock::now();

    double total_us =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            iter_end - iter_start)
            .count();
    total_times.push_back(total_us);
    compute_times.push_back(compute_us);
    comm_times.push_back(comm_us);
  }

  profile.total = summarize(total_times);
  profile.compute = summarize(compute_times);
  profile.comm = summarize(comm_times);
  return profile;
}

// Comm-only benchmark: measures pure AllReduce latency without compute
Stats benchmark_comm_only(const mx::array& tensor,
                          const mx::distributed::Group& group,
                          const Args& args,
                          bool verbose) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = tensor.size() * elem_size;

  if (verbose) {
    std::cout << "[comm-only] Tensor shape: [" << args.batch_size << ", "
              << args.hidden_size << "]\n";
    std::cout << "[comm-only] Tensor size: " << tensor_bytes << " bytes ("
              << tensor_bytes / 1024.0 << " KB)\n";
    std::cout << "[comm-only] Warmup iterations: " << args.warmup << std::endl;
  }

  // Warmup
  for (int i = 0; i < args.warmup; ++i) {
    auto result = mx::distributed::all_sum(tensor, group);
    mx::eval(result);
  }

  std::vector<double> times;
  times.reserve(args.iterations);

  if (verbose) {
    std::cout << "[comm-only] Timed iterations: " << args.iterations
              << std::endl;
  }

  for (int i = 0; i < args.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = mx::distributed::all_sum(tensor, group);
    mx::eval(result);
    auto end = std::chrono::high_resolution_clock::now();
    double usec =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            end - start)
            .count();
    times.push_back(usec);
  }

  return summarize(times);
}

// Fake compute benchmark: measures AllReduce latency with zero-copy tensors
// (no actual GPU compute, just communication)
Stats benchmark_fake_compute(const std::vector<BlockWeights>& blocks,
                             const mx::array& input,
                             const mx::distributed::Group& group,
                             const Args& args,
                             bool verbose) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = input.size() * elem_size;

  if (verbose) {
    std::cout << "[fake-compute] Tensor shape: [" << args.batch_size << ", "
              << args.hidden_size << "]\n";
    std::cout << "[fake-compute] Tensor size: " << tensor_bytes << " bytes ("
              << tensor_bytes / 1024.0 << " KB)\n";
    std::cout << "[fake-compute] Num blocks (AllReduce calls): " << args.num_blocks
              << "\n";
    std::cout << "[fake-compute] Warmup iterations: " << args.warmup << std::endl;
  }

  // Warmup
  for (int i = 0; i < args.warmup; ++i) {
    auto out = apply_blocks_fake(input, blocks, group);
    mx::eval(out);
  }

  std::vector<double> times;
  times.reserve(args.iterations);

  if (verbose) {
    std::cout << "[fake-compute] Timed iterations: " << args.iterations
              << std::endl;
  }

  for (int i = 0; i < args.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto out = apply_blocks_fake(input, blocks, group);
    mx::eval(out);
    auto end = std::chrono::high_resolution_clock::now();
    double usec =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            end - start)
            .count();
    times.push_back(usec);
  }

  return summarize(times);
}

void print_fake_compute_stats(const Stats& stats,
                              const mx::distributed::Group& group,
                              const Args& args) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = args.batch_size * args.hidden_size * elem_size;

  std::cout << "\n" << std::string(70, '=') << "\n"
            << "Fake Compute Benchmark (rank " << group.rank() << "/"
            << group.size() << ")\n"
            << std::string(70, '=') << std::endl;

  std::cout << "\n*** No GPU compute - only AllReduce with zero tensors ***\n\n";

  std::cout << "Configuration:\n"
            << "  Hidden size:    " << args.hidden_size << "\n"
            << "  Batch size:     " << args.batch_size << "\n"
            << "  Tensor bytes:   " << tensor_bytes << " ("
            << tensor_bytes / 1024.0 << " KB)\n"
            << "  Num blocks:     " << args.num_blocks << " (AllReduce calls per iter)\n"
            << "  World size:     " << group.size() << "\n"
            << "  Iterations:     " << args.iterations << "\n\n";

  std::cout << "Total Iteration Latency:\n"
            << "  Mean:  " << stats.mean << " μs (" << stats.mean / 1000.0 << " ms)\n"
            << "  Min:   " << stats.min << " μs\n"
            << "  p50:   " << stats.p50 << " μs\n"
            << "  p95:   " << stats.p95 << " μs\n"
            << "  p99:   " << stats.p99 << " μs\n"
            << "  Max:   " << stats.max << " μs\n\n";

  // Per-AllReduce stats
  double per_allreduce = stats.mean / args.num_blocks;
  std::cout << "Per-AllReduce Latency:\n"
            << "  Mean:  " << per_allreduce << " μs\n\n";

  // Calculate bandwidth
  double bandwidth_gbps =
      (tensor_bytes * args.num_blocks / (stats.mean / 1e6)) /
      (1024.0 * 1024.0 * 1024.0);
  std::cout << "Effective Bandwidth:\n"
            << "  " << bandwidth_gbps << " GB/s (aggregate)\n";
}

// Send/Recv benchmark: measures point-to-point latency using explicit send/recv
// This bypasses AllReduce to measure raw JACCL send/recv performance
Stats benchmark_send_recv(const mx::array& tensor,
                          const mx::distributed::Group& group,
                          const Args& args,
                          bool verbose) {
  if (group.size() != 2) {
    throw std::runtime_error("--send-recv requires exactly 2 ranks");
  }

  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = tensor.size() * elem_size;
  int my_rank = group.rank();
  int peer_rank = (my_rank == 0) ? 1 : 0;

  if (verbose) {
    std::cout << "[send-recv] Rank " << my_rank << " <-> Rank " << peer_rank << "\n";
    std::cout << "[send-recv] Tensor shape: [" << args.batch_size << ", "
              << args.hidden_size << "]\n";
    std::cout << "[send-recv] Tensor size: " << tensor_bytes << " bytes ("
              << tensor_bytes / 1024.0 << " KB)\n";
    std::cout << "[send-recv] Num round-trips per iter: " << args.num_blocks << "\n";
    std::cout << "[send-recv] Warmup iterations: " << args.warmup << std::endl;
  }

  // Create a receive buffer
  auto recv_buf = mx::zeros_like(tensor);
  mx::eval(recv_buf);

  // Warmup: each rank sends then receives (ping-pong)
  for (int i = 0; i < args.warmup; ++i) {
    for (int b = 0; b < args.num_blocks; ++b) {
      if (my_rank == 0) {
        // Rank 0: send then recv
        auto send_token = mx::distributed::send(tensor, peer_rank, group);
        mx::eval(send_token);
        recv_buf = mx::distributed::recv_like(tensor, peer_rank, group);
        mx::eval(recv_buf);
      } else {
        // Rank 1: recv then send
        recv_buf = mx::distributed::recv_like(tensor, peer_rank, group);
        mx::eval(recv_buf);
        auto send_token = mx::distributed::send(tensor, peer_rank, group);
        mx::eval(send_token);
      }
    }
  }

  std::vector<double> times;
  times.reserve(args.iterations);

  if (verbose) {
    std::cout << "[send-recv] Timed iterations: " << args.iterations << std::endl;
  }

  for (int i = 0; i < args.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int b = 0; b < args.num_blocks; ++b) {
      if (my_rank == 0) {
        auto send_token = mx::distributed::send(tensor, peer_rank, group);
        mx::eval(send_token);
        recv_buf = mx::distributed::recv_like(tensor, peer_rank, group);
        mx::eval(recv_buf);
      } else {
        recv_buf = mx::distributed::recv_like(tensor, peer_rank, group);
        mx::eval(recv_buf);
        auto send_token = mx::distributed::send(tensor, peer_rank, group);
        mx::eval(send_token);
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double usec =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            end - start)
            .count();
    times.push_back(usec);
  }

  return summarize(times);
}

void print_send_recv_stats(const Stats& stats,
                           const mx::distributed::Group& group,
                           const Args& args) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = args.batch_size * args.hidden_size * elem_size;

  std::cout << "\n" << std::string(70, '=') << "\n"
            << "Send/Recv Benchmark (rank " << group.rank() << "/"
            << group.size() << ")\n"
            << std::string(70, '=') << std::endl;

  std::cout << "\n*** Point-to-point send/recv (ping-pong) ***\n\n";

  std::cout << "Configuration:\n"
            << "  Hidden size:      " << args.hidden_size << "\n"
            << "  Batch size:       " << args.batch_size << "\n"
            << "  Tensor bytes:     " << tensor_bytes << " ("
            << tensor_bytes / 1024.0 << " KB)\n"
            << "  Round-trips/iter: " << args.num_blocks << "\n"
            << "  World size:       " << group.size() << "\n"
            << "  Iterations:       " << args.iterations << "\n\n";

  std::cout << "Total Iteration Latency:\n"
            << "  Mean:  " << stats.mean << " μs (" << stats.mean / 1000.0 << " ms)\n"
            << "  Min:   " << stats.min << " μs\n"
            << "  p50:   " << stats.p50 << " μs\n"
            << "  p95:   " << stats.p95 << " μs\n"
            << "  p99:   " << stats.p99 << " μs\n"
            << "  Max:   " << stats.max << " μs\n\n";

  // Per round-trip (send + recv)
  double per_roundtrip = stats.mean / args.num_blocks;
  // Per one-way message (send OR recv)
  double per_message = per_roundtrip / 2.0;

  std::cout << "Per-Operation Latency:\n"
            << "  Per round-trip (send+recv): " << per_roundtrip << " μs\n"
            << "  Per one-way message:        " << per_message << " μs\n\n";

  // Calculate bandwidth (one-way)
  double bandwidth_gbps =
      (tensor_bytes / (per_message / 1e6)) / (1024.0 * 1024.0 * 1024.0);
  std::cout << "Effective Bandwidth (one-way):\n"
            << "  " << bandwidth_gbps << " GB/s\n";
}

// Custom AllReduce using send/recv (for 2 ranks)
// Uses ping-pong pattern: rank 0 sends first, rank 1 recvs first to avoid deadlock
// Minimizes evals: 2 evals per AllReduce (one for send, one for recv)
mx::array custom_all_sum(const mx::array& input,
                         const mx::distributed::Group& group) {
  if (group.size() != 2) {
    throw std::runtime_error("custom_all_sum requires exactly 2 ranks");
  }

  int my_rank = group.rank();
  int peer_rank = (my_rank == 0) ? 1 : 0;

  if (my_rank == 0) {
    // Rank 0: send first, then recv
    auto send_token = mx::distributed::send(input, peer_rank, group);
    mx::eval(send_token);
    auto received = mx::distributed::recv_like(input, peer_rank, group);
    mx::eval(received);
    return mx::add(input, received);
  } else {
    // Rank 1: recv first, then send
    auto received = mx::distributed::recv_like(input, peer_rank, group);
    mx::eval(received);
    auto send_token = mx::distributed::send(input, peer_rank, group);
    mx::eval(send_token);
    return mx::add(input, received);
  }
}

// Apply blocks using custom AllReduce instead of mx::distributed::all_sum
mx::array apply_blocks_custom_allreduce(const mx::array& input,
                                        const std::vector<BlockWeights>& blocks,
                                        const mx::distributed::Group& group) {
  mx::array current = input;
  for (const auto& block : blocks) {
    auto up = mx::matmul(current, mx::transpose(block.w_up));
    auto activated = gelu(up);
    auto partial = mx::matmul(activated, mx::transpose(block.w_down));
    if (group.size() > 1) {
      current = custom_all_sum(partial, group);
    } else {
      current = partial;
    }
  }
  return current;
}

// Benchmark custom AllReduce implementation
Stats benchmark_custom_allreduce(const std::vector<BlockWeights>& blocks,
                                 const mx::array& input,
                                 const mx::distributed::Group& group,
                                 const Args& args,
                                 bool verbose) {
  if (group.size() != 2) {
    throw std::runtime_error("--custom-allreduce requires exactly 2 ranks");
  }

  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = args.batch_size * args.hidden_size * elem_size;

  if (verbose) {
    std::cout << "[custom-allreduce] Tensor shape: [" << args.batch_size << ", "
              << args.hidden_size << "]\n";
    std::cout << "[custom-allreduce] Tensor size: " << tensor_bytes << " bytes ("
              << tensor_bytes / 1024.0 << " KB)\n";
    std::cout << "[custom-allreduce] Num blocks: " << args.num_blocks << "\n";
    std::cout << "[custom-allreduce] Warmup iterations: " << args.warmup << std::endl;
  }

  // Warmup
  for (int i = 0; i < args.warmup; ++i) {
    auto out = apply_blocks_custom_allreduce(input, blocks, group);
    mx::eval(out);
  }

  std::vector<double> times;
  times.reserve(args.iterations);

  if (verbose) {
    std::cout << "[custom-allreduce] Timed iterations: " << args.iterations
              << std::endl;
  }

  for (int i = 0; i < args.iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    auto out = apply_blocks_custom_allreduce(input, blocks, group);
    mx::eval(out);
    auto end = std::chrono::high_resolution_clock::now();
    double usec =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            end - start)
            .count();
    times.push_back(usec);
  }

  return summarize(times);
}

void print_custom_allreduce_stats(const Stats& stats,
                                  const mx::distributed::Group& group,
                                  const Args& args) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = args.batch_size * args.hidden_size * elem_size;

  std::cout << "\n" << std::string(70, '=') << "\n"
            << "Custom AllReduce Benchmark (rank " << group.rank() << "/"
            << group.size() << ")\n"
            << std::string(70, '=') << std::endl;

  std::cout << "\n*** Using send/recv instead of all_sum primitive ***\n\n";

  std::cout << "Configuration:\n"
            << "  Hidden size:       " << args.hidden_size << "\n"
            << "  Intermediate size: " << args.intermediate_size << "\n"
            << "  Batch size:        " << args.batch_size << "\n"
            << "  Tensor bytes:      " << tensor_bytes << " ("
            << tensor_bytes / 1024.0 << " KB)\n"
            << "  Num blocks:        " << args.num_blocks << "\n"
            << "  World size:        " << group.size() << "\n"
            << "  Iterations:        " << args.iterations << "\n\n";

  std::cout << "Total Iteration Latency:\n"
            << "  Mean:  " << stats.mean << " μs (" << stats.mean / 1000.0 << " ms)\n"
            << "  Min:   " << stats.min << " μs\n"
            << "  p50:   " << stats.p50 << " μs\n"
            << "  p95:   " << stats.p95 << " μs\n"
            << "  p99:   " << stats.p99 << " μs\n"
            << "  Max:   " << stats.max << " μs\n\n";

  double per_block = stats.mean / args.num_blocks;
  std::cout << "Per-Block Latency:\n"
            << "  Mean:  " << per_block << " μs\n";
}

void print_comm_only_stats(const Stats& stats,
                           const mx::distributed::Group& group,
                           const Args& args) {
  size_t elem_size = (args.dtype == "float32") ? 4 : 2;
  size_t tensor_bytes = args.batch_size * args.hidden_size * elem_size;

  std::cout << "\n" << std::string(70, '=') << "\n"
            << "Communication-Only Benchmark (rank " << group.rank() << "/"
            << group.size() << ")\n"
            << std::string(70, '=') << std::endl;

  std::cout << "\nConfiguration:\n"
            << "  Hidden size:    " << args.hidden_size << "\n"
            << "  Batch size:     " << args.batch_size << "\n"
            << "  Tensor bytes:   " << tensor_bytes << " ("
            << tensor_bytes / 1024.0 << " KB)\n"
            << "  World size:     " << group.size() << "\n"
            << "  Iterations:     " << args.iterations << "\n\n";

  std::cout << "AllReduce Latency:\n"
            << "  Mean:  " << stats.mean << " μs\n"
            << "  Min:   " << stats.min << " μs\n"
            << "  p50:   " << stats.p50 << " μs\n"
            << "  p95:   " << stats.p95 << " μs\n"
            << "  p99:   " << stats.p99 << " μs\n"
            << "  Max:   " << stats.max << " μs\n\n";

  // Calculate bandwidth
  // For all_sum with 2 ranks: each rank sends tensor_bytes, receives tensor_bytes
  double bandwidth_gbps =
      (tensor_bytes / (stats.mean / 1e6)) / (1024.0 * 1024.0 * 1024.0);
  std::cout << "Effective Bandwidth:\n"
            << "  " << bandwidth_gbps << " GB/s (unidirectional)\n"
            << "  " << bandwidth_gbps * 2 << " GB/s (bidirectional)\n";
}

void print_comm_profile(const CommProfile& profile,
                        const mx::distributed::Group& group,
                        const Args& args) {
  std::cout << "\n" << std::string(70, '=') << "\n"
            << "Communication Profile (rank " << group.rank() << "/"
            << group.size() << ")\n"
            << std::string(70, '=') << std::endl;

  std::cout << "\n*** WARNING: Profiling mode inserts sync points after each op ***\n"
            << "*** This disrupts GPU pipelining - times are NOT representative ***\n"
            << "*** of actual performance. Use for relative comm/compute ratio. ***\n\n";

  std::cout << "Configuration:\n"
            << "  Hidden size:       " << args.hidden_size << "\n"
            << "  Intermediate size: " << args.intermediate_size << "\n"
            << "  Batch size:        " << args.batch_size << "\n"
            << "  Num blocks:        " << args.num_blocks << "\n"
            << "  World size:        " << group.size() << "\n\n";

  double total_mean = profile.total.mean;
  double compute_mean = profile.compute.mean;
  double comm_mean = profile.comm.mean;
  double compute_pct = (compute_mean / total_mean) * 100.0;
  double comm_pct = (comm_mean / total_mean) * 100.0;
  double overhead_pct = 100.0 - compute_pct - comm_pct;

  std::cout << "Timing Breakdown (mean per iteration):\n"
            << "  Total:    " << total_mean << " μs (" << total_mean / 1000.0
            << " ms)\n"
            << "  Compute:  " << compute_mean << " μs (" << compute_mean / 1000.0
            << " ms) [" << compute_pct << "%]\n"
            << "  Comm:     " << comm_mean << " μs (" << comm_mean / 1000.0
            << " ms) [" << comm_pct << "%]\n"
            << "  Overhead: " << (total_mean - compute_mean - comm_mean)
            << " μs [" << overhead_pct << "%]\n\n";

  std::cout << "Per-AllReduce Statistics:\n"
            << "  AllReduce calls/iter: " << profile.num_allreduce_calls << "\n"
            << "  Bytes per AllReduce:  " << profile.bytes_per_allreduce
            << " (" << profile.bytes_per_allreduce / 1024.0 << " KB)\n";

  double per_allreduce_us = comm_mean / profile.num_allreduce_calls;
  std::cout << "  Mean time/AllReduce:  " << per_allreduce_us << " μs\n";

  // Calculate effective bandwidth (bidirectional for all_sum)
  // all_sum: each rank sends N bytes, receives N*(world_size-1) bytes
  double total_bytes_moved =
      profile.bytes_per_allreduce * profile.num_allreduce_calls;
  double bandwidth_gbps =
      (total_bytes_moved / (comm_mean / 1e6)) / (1024.0 * 1024.0 * 1024.0);
  std::cout << "  Effective bandwidth:  " << bandwidth_gbps << " GB/s\n\n";

  std::cout << "Detailed Stats:\n";
  std::cout << "  Total  - min: " << profile.total.min
            << " μs, p50: " << profile.total.p50
            << " μs, p99: " << profile.total.p99
            << " μs, max: " << profile.total.max << " μs\n";
  std::cout << "  Compute - min: " << profile.compute.min
            << " μs, p50: " << profile.compute.p50
            << " μs, p99: " << profile.compute.p99
            << " μs, max: " << profile.compute.max << " μs\n";
  std::cout << "  Comm   - min: " << profile.comm.min
            << " μs, p50: " << profile.comm.p50
            << " μs, p99: " << profile.comm.p99
            << " μs, max: " << profile.comm.max << " μs\n";
}

void print_stats(const std::string& label,
                 const Stats& stats,
                 const Args& args) {
  std::cout << "\n" << std::string(70, '=') << "\n" << label << "\n"
            << std::string(70, '=') << std::endl;
  std::cout << "Hidden size: " << args.hidden_size << "\n"
            << "Intermediate size: " << args.intermediate_size << "\n"
            << "Batch size: " << args.batch_size << "\n"
            << "Num blocks: " << args.num_blocks << "\n\n";
  std::cout << "Mean: " << stats.mean << " μs (" << stats.mean / 1000.0
            << " ms)\n";
  std::cout << "Min:  " << stats.min << " μs\n";
  std::cout << "p50:  " << stats.p50 << " μs\n";
  std::cout << "p95:  " << stats.p95 << " μs\n";
  std::cout << "p99:  " << stats.p99 << " μs\n";
  std::cout << "Max:  " << stats.max << " μs\n";
}

std::string stats_json(const Stats& stats) {
  std::ostringstream oss;
  oss << "{\"mean\":" << stats.mean << ",\"min\":" << stats.min
      << ",\"max\":" << stats.max << ",\"p50\":" << stats.p50
      << ",\"p95\":" << stats.p95 << ",\"p99\":" << stats.p99 << "}";
  return oss.str();
}

std::string self_command(const Args& args,
                         const std::string& exec_path,
                         const std::optional<std::string>& trace_override) {
  std::ostringstream cmd;
  cmd << "\"" << exec_path << "\""
      << " --hidden-size " << args.hidden_size
      << " --intermediate-size " << args.intermediate_size << " --batch-size "
      << args.batch_size << " --iterations " << args.iterations << " --warmup "
      << args.warmup << " --num-blocks " << args.num_blocks << " --dtype "
      << args.dtype << " --single-host --skip-single --emit-json"
      << " --single-host-runner " << args.single_host_runner;
  if (trace_override.has_value()) {
    cmd << " --metal-trace \"" << *trace_override << "\"";
  }
  if (args.cpu_only) {
    cmd << " --cpu";
  }
  return cmd.str();
}

std::string run_command_capture(const std::string& command) {
  std::array<char, 4096> buffer{};
  std::string output;
  std::string redirected = command + " 2>&1";
  FILE* pipe = popen(redirected.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("Failed to run command: " + command);
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
    output.append(buffer.data());
  }
  int rc = pclose(pipe);
  if (rc != 0) {
    std::ostringstream oss;
    oss << "Command failed with code " << rc << ": " << command;
    throw std::runtime_error(oss.str());
  }
  return output;
}

std::optional<Stats> parse_stats_from_output(const std::string& output) {
  const std::string tag = "RESULT_JSON:";
  auto pos = output.rfind(tag);
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  std::string json = output.substr(pos + tag.size());
  auto parse_field = [&](const std::string& key) -> double {
    std::string pattern = "\"" + key + "\":";
    auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
      throw std::runtime_error("Missing field in RESULT_JSON: " + key);
    }
    key_pos += pattern.size();
    size_t end = key_pos;
    while (end < json.size() &&
           (std::isdigit(json[end]) || json[end] == '-' || json[end] == '+' ||
            json[end] == '.' || json[end] == 'e' || json[end] == 'E')) {
      ++end;
    }
    return std::stod(json.substr(key_pos, end - key_pos));
  };
  Stats stats;
  stats.mean = parse_field("mean");
  stats.min = parse_field("min");
  stats.max = parse_field("max");
  stats.p50 = parse_field("p50");
  stats.p95 = parse_field("p95");
  stats.p99 = parse_field("p99");
  return stats;
}

void verify_correctness(const std::vector<BlockWeights>& blocks,
                        const mx::array& input,
                        const mx::distributed::Group& group,
                        const Args& args,
                        bool verbose) {
  if (args.single_host || group.size() == 1) {
    return;
  }
  if (verbose && group.rank() == 0) {
    std::cout << "[verbose] Running TP vs single-host correctness check..."
              << std::endl;
  }
  auto tp_out = apply_blocks(input, blocks, group);
  mx::eval(tp_out);
  bool keep_data = group.rank() == 0;
  auto full_blocks = gather_full_blocks(blocks, group, keep_data);
  if (!keep_data) {
    return;
  }
  auto single_out = apply_blocks_local(input, full_blocks);
  mx::eval(single_out);
  auto diff =
      mx::abs(mx::astype(tp_out, mx::float32) -
              mx::astype(single_out, mx::float32));
  double max_diff = mx::max(diff).item<double>();
  if (max_diff > 1e-3) {
    std::ostringstream oss;
    oss << "TP vs single-host mismatch max_diff=" << max_diff;
    throw std::runtime_error(oss.str());
  }
  if (group.rank() == 0) {
    std::cout << "Correctness check passed (max diff " << max_diff << ")\n";
  }
}

int main(int argc, char** argv) {
  // Enable fast Metal sync (spin-wait instead of MTL::SharedEvent)
  // Must be set before any MLX initialization
  setenv("MLX_METAL_FAST_SYNCH", "1", 1);

  try {
    Args args = parse_args(argc, argv);
    std::cout << "separate_tp_vs_single benchmark v" << kBenchmarkVersion
              << " (MLX_METAL_FAST_SYNCH=1)" << std::endl;
    bool verbose_logging = args.verbose_env.has_value();
    if (verbose_logging) {
      std::cout << "[verbose] hidden=" << args.hidden_size
                << " intermediate=" << args.intermediate_size
                << " batch=" << args.batch_size
                << " blocks=" << args.num_blocks
                << " iterations=" << args.iterations
                << " warmup=" << args.warmup
                << (args.single_host ? " (single host)" : " (tensor parallel)")
                << std::endl;
    }
    mx::random::seed(42);
    auto dtype = parse_dtype(args.dtype);

    // Set device (CPU or GPU)
    if (args.cpu_only) {
      mx::set_default_device(mx::Device::cpu);
      if (verbose_logging) {
        std::cout << "[verbose] Using CPU backend\n";
      }
    } else {
      mx::set_default_device(mx::Device::gpu);
      if (verbose_logging) {
        std::cout << "[verbose] Using GPU backend\n";
      }
    }

    auto apply_env = [](const char* key,
                        const std::optional<std::string>& value) {
      if (value.has_value()) {
        setenv(key, value->c_str(), 1);
      }
    };
    apply_env("MLX_RANK", args.rank_env);
    apply_env("MLX_IBV_COORDINATOR", args.coordinator_env);
    apply_env("MLX_IBV_DEVICES", args.devices_env);
    apply_env("MLX_IBV_VERBOSE", args.verbose_env);

    // Run sync benchmark if requested (local only, no distributed)
    if (args.sync_bench) {
      benchmark_sync(args, verbose_logging);
      return 0;
    }

    mx::array input =
        mx::random::normal({args.batch_size, args.hidden_size}, dtype);
    mx::eval(input);

    auto group = args.single_host ? mx::distributed::init()
                                  : mx::distributed::init(true, "jaccl");

    // Run comm-only benchmark if requested (no compute, just AllReduce)
    if (args.comm_only) {
      auto comm_stats = benchmark_comm_only(input, group, args, verbose_logging);
      print_comm_only_stats(comm_stats, group, args);
      return 0;
    }

    // Run send/recv benchmark if requested (point-to-point, no AllReduce)
    if (args.send_recv) {
      auto sr_stats = benchmark_send_recv(input, group, args, verbose_logging);
      print_send_recv_stats(sr_stats, group, args);
      return 0;
    }

    auto blocks = create_blocks(args, group, dtype);

    // Run custom AllReduce benchmark (using send/recv instead of all_sum)
    if (args.custom_allreduce) {
      auto custom_stats =
          benchmark_custom_allreduce(blocks, input, group, args, verbose_logging);
      print_custom_allreduce_stats(custom_stats, group, args);

      // Also run native all_sum for comparison
      if (verbose_logging) {
        std::cout << "\n[custom-allreduce] Running native all_sum for comparison..."
                  << std::endl;
      }
      auto native_stats = benchmark(blocks, input, group, args, verbose_logging);

      std::cout << "\nComparison (Custom vs Native AllReduce):\n"
                << "  Custom (send/recv): " << custom_stats.mean << " μs ("
                << custom_stats.mean / 1000.0 << " ms)\n"
                << "  Native (all_sum):   " << native_stats.mean << " μs ("
                << native_stats.mean / 1000.0 << " ms)\n"
                << "  Speedup:            "
                << native_stats.mean / custom_stats.mean << "x\n";
      return 0;
    }

    // Run hybrid benchmark: GPU compute + CPU AllReduce
    if (args.hybrid) {
      auto hybrid_stats =
          benchmark_hybrid(blocks, input, group, args, verbose_logging);
      print_hybrid_stats(hybrid_stats, group, args);

      // Also run standard GPU benchmark for comparison
      if (verbose_logging) {
        std::cout << "\n[hybrid] Running standard GPU benchmark for comparison..."
                  << std::endl;
      }
      auto gpu_stats = benchmark(blocks, input, group, args, verbose_logging);

      std::cout << "\nComparison (Hybrid vs Standard GPU):\n"
                << "  Hybrid (GPU+CPU):   " << hybrid_stats.mean << " μs ("
                << hybrid_stats.mean / 1000.0 << " ms)\n"
                << "  Standard (GPU):     " << gpu_stats.mean << " μs ("
                << gpu_stats.mean / 1000.0 << " ms)\n"
                << "  Speedup:            "
                << gpu_stats.mean / hybrid_stats.mean << "x\n";
      return 0;
    }

    // Run fake-compute benchmark if requested (zeros for AllReduce, no GPU compute)
    // This can be run on one rank while the other runs normal compute to test
    // asymmetric workloads and AllReduce behavior with fast/slow ranks
    if (args.fake_compute) {
      auto fake_stats =
          benchmark_fake_compute(blocks, input, group, args, verbose_logging);
      print_fake_compute_stats(fake_stats, group, args);
      return 0;
    }

    // Run communication profiling if requested
    if (args.profile_comms) {
      // Run profiled benchmark (with sync points)
      auto profile = benchmark_comms(blocks, input, group, args, verbose_logging);
      print_comm_profile(profile, group, args);

      // Barrier to ensure all ranks complete profiling before normal benchmark
      auto barrier = mx::distributed::all_sum(mx::zeros({1}, mx::float32), group);
      mx::eval(barrier);

      // Now run normal benchmark to get pipelined time for comparison
      if (verbose_logging) {
        std::cout << "[profile-comms] Running pipelined benchmark for comparison..."
                  << std::endl;
      }
      auto pipelined_stats = benchmark(blocks, input, group, args, verbose_logging);

      // Show comparison with pipelined execution
      std::cout << "\nPipeline Efficiency Comparison:\n"
                << "  Pipelined (normal):  " << pipelined_stats.mean << " μs ("
                << pipelined_stats.mean / 1000.0 << " ms)\n"
                << "  Profiled (sync'd):   " << profile.total.mean << " μs ("
                << profile.total.mean / 1000.0 << " ms)\n"
                << "  Pipeline speedup:    "
                << profile.total.mean / pipelined_stats.mean << "x\n"
                << "  Sync overhead:       "
                << (profile.total.mean - pipelined_stats.mean) << " μs ("
                << ((profile.total.mean / pipelined_stats.mean) - 1.0) * 100.0
                << "%)\n\n";

      // Estimate actual comm time in pipelined mode
      double comm_ratio = profile.comm.mean / profile.total.mean;
      double estimated_comm = pipelined_stats.mean * comm_ratio;
      std::cout << "Estimated Pipelined Breakdown:\n"
                << "  Est. compute: " << (pipelined_stats.mean - estimated_comm)
                << " μs (" << (1.0 - comm_ratio) * 100.0 << "%)\n"
                << "  Est. comm:    " << estimated_comm << " μs ("
                << comm_ratio * 100.0 << "%)\n";
      return 0;
    }

    auto stats = benchmark(blocks, input, group, args, verbose_logging);
    std::ostringstream label;
    if (args.single_host) {
      label << "Single Host Separate";
    } else {
      label << "Tensor Parallel Separate (rank " << group.rank() << "/"
            << group.size() << ")";
    }
    print_stats(label.str(), stats, args);
    if (args.emit_json) {
      std::cout << "RESULT_JSON:" << stats_json(stats) << std::endl;
    }

    if (!args.single_host) {
      verify_correctness(blocks, input, group, args, verbose_logging);
    }

    if (!args.single_host && !args.skip_single &&
        group.rank() == args.single_host_runner) {
      if (verbose_logging) {
        std::cout << "[verbose] Launching single-host comparison run"
                  << std::endl;
      } else {
        std::cout << "\nLaunching single-host comparison run...\n";
      }
      std::cout << std::endl;
      auto exec_path = std::filesystem::canonical(argv[0]).string();
      std::optional<std::string> single_trace;
      if (args.metal_trace.has_value()) {
        single_trace = *args.metal_trace + "_single";
      }
      auto cmd = self_command(args, exec_path, single_trace);
      static const char* kUnsetKeys[] = {"MLX_RANK",
                                         "MLX_IBV_COORDINATOR",
                                         "MLX_IBV_DEVICES",
                                         "MLX_IBV_VERBOSE"};
      std::ostringstream env_cmd;
      env_cmd << "env";
      for (const char* key : kUnsetKeys) {
        env_cmd << " -u " << key;
      }
      env_cmd << " " << cmd;
      auto output = run_command_capture(env_cmd.str());
      std::cout << output << std::flush;
      auto single_stats = parse_stats_from_output(output);
      if (single_stats.has_value()) {
        std::cout << "\n===== TP vs Single-Host Comparison =====\n";
        std::cout << "TP mean:     " << stats.mean << " μs ("
                  << stats.mean / 1000.0 << " ms)\n";
        std::cout << "Single mean: " << single_stats->mean << " μs ("
                  << single_stats->mean / 1000.0 << " ms)\n";
        std::cout << "Speedup:     "
                  << single_stats->mean / stats.mean << "x\n";
      } else {
        std::cout << "WARNING: Could not parse single-host stats for "
                     "summary.\n";
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
