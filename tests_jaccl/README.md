# JACCL Distributed MLP Test

This directory contains scripts to test Tensor Parallel MLP using the JACCL backend on `m4p.local` and `m3u.local`.

## Configuration

- **Hosts**:
  - Rank 0: `m4p.local` (IP: `10.1.12.4`, Device: `rdma_en2`)
  - Rank 1: `m3u.local` (IP: `10.1.12.3`, Device: `rdma_en5`)
- **Config File**: `jaccl_config.json` maps ranks to devices.

## Files

- `test_tp_mlp.py`: The distributed MLP test script.
- `jaccl_config.json`: Device configuration.
- `run_jaccl.sh`: Helper script to run the test.
- `deploy.sh`: Helper script to sync files to hosts.

## How to Run

1.  **Deploy files**:
    Run `./deploy.sh` from this directory to sync code to both hosts.

2.  **Run on Rank 0 (m4p.local)**:
    ```bash
    ssh m4p.local
    cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma/tests_jaccl
    ./run_jaccl.sh 0 10.1.12.4:1234
    ```

3.  **Run on Rank 1 (m3u.local)**:
    ```bash
    ssh m3u.local
    cd /Users/anemll/SourceRelease/GITHUB/ML_playground/mlx-rdma/tests_jaccl
    ./run_jaccl.sh 1 10.1.12.4:1234
    ```
