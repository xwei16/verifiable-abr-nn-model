import os
import random
import numpy as np
import ppo2 as network

def row_to_state_no_change(r):
    """Convert one pandas row to Pensieve state (6×8)."""
    s = np.zeros((6, 8), dtype=np.float32)
    s[0, -1] = r["Last1_chunk_bitrate"]
    s[1, -1] = r["Last1_buffer_size"]
    s[2, -1] = r["Last1_throughput"]
    s[3, -1] = r["Last1_downloadtime"]
    s[4, :6] = r[[f"chunksize{i}"        for i in range(1, 7)]].values
    s[5, -1] = r["Chunks_left"]

    return s


# TODO: hardcoded
def row_to_state(r, spec):
    """Convert one pandas row to Pensieve state (6×8)."""
    s = np.zeros((6, 8), dtype=np.float32)
    s[0, 0] = r["Last1_chunk_bitrate"]
    s[1, 0] = r["Last1_buffer_size"]
    s[2] = r[[f"Last{i}_throughput"   for i in range(8, 0, -1)]].values
    s[3] = r[[f"Last{i}_downloadtime" for i in range(8, 0, -1)]].values
    s[4, :6] = r[[f"chunksize{i}"        for i in range(1, 7)]].values
    s[5, 0] = r["Chunks_left"]

    chosen = spec[random.randint(0, len(spec) - 1)]   
    s[3, 7] = random.uniform(
        chosen["Last1_downloadtime_l"],
        chosen["Last1_downloadtime_u"]
    )
    low = chosen["Last1_downloadtime_l"]
    high = chosen["Last1_downloadtime_u"]

    return s, low, high

# XXX: ppo2
def load_ppo2_model(model_path, state_dim, action_dim, learning_rate):
    if os.path.exists(model_path):
        # TODO: load parameters from file
        actor = network.Network(state_dim=state_dim,
                                action_dim=action_dim,
                                learning_rate=learning_rate)
        model = actor.load_model(model_path)
        model = actor
        # TODO: move to GPU
        # model.to(DEVICE)
        print(model)

        return model
    else:
        raise FileNotFoundError(f"PyTorch model not found at {model_path}")
    