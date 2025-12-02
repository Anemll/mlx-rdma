// Copyright © 2025 Apple Inc.

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
constexpr const char* kBenchmarkVersion = "0.0.4";

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
};

struct Stats {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
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
  try {
    Args args = parse_args(argc, argv);
    std::cout << "separate_tp_vs_single benchmark v" << kBenchmarkVersion
              << std::endl;
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

    mx::array input =
        mx::random::normal({args.batch_size, args.hidden_size}, dtype);
    mx::eval(input);

    auto group = args.single_host ? mx::distributed::init()
                                  : mx::distributed::init(true, "jaccl");
    auto blocks = create_blocks(args, group, dtype);

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
