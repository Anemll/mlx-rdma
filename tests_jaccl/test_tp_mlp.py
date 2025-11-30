import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

# Access distributed through mx.core.distributed
dist = mx.distributed

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()

    def __call__(self, x):
        return self.l2(self.gelu(self.l1(x)))

def main():
    # Initialize distributed backend
    # strict=True ensures we fail if env vars are missing
    world = dist.init(strict=True, backend="jaccl")
    rank = world.rank()
    size = world.size()
    
    print(f"Rank {rank}/{size} initialized")

    # Dimensions
    D = 16
    H = 32
    
    # Create model
    # Note: In a real scenario, you might want to initialize weights identically on all ranks
    # or load from a checkpoint. Here we rely on seeding.
    mx.random.seed(42)
    model = MLP(D, H, D)
    
    # Shard model for Tensor Parallelism
    # Layer 1: Column Parallel (AllToSharded)
    # Input is replicated (All), Output is sharded
    model.l1 = shard_linear(model.l1, "all-to-sharded", group=world)
    
    # Layer 2: Row Parallel (ShardedToAll)
    # Input is sharded, Output is replicated (All)
    model.l2 = shard_linear(model.l2, "sharded-to-all", group=world)
    
    # Input data
    # Input should be replicated (same on all ranks)
    mx.random.seed(123)
    x = mx.random.normal((4, D))
    
    # Forward pass
    y = model(x)
    mx.eval(y)
    
    print(f"Rank {rank}: Output shape {y.shape}")
    print(f"Rank {rank}: Output sample {y[0, :4]}")
    
    # Verification (Optional: compare with local model if running on rank 0)
    if rank == 0:
        print("Rank 0: Verifying against local model...")
        mx.random.seed(42)
        local_model = MLP(D, H, D)
        mx.random.seed(123)
        local_x = mx.random.normal((4, D))
        local_y = local_model(local_x)
        
        if mx.allclose(y, local_y, atol=1e-5):
            print("SUCCESS: Distributed output matches local output.")
        else:
            print("FAILURE: Distributed output does NOT match local output.")
            print(f"Local sample: {local_y[0, :4]}")

if __name__ == "__main__":
    main()
