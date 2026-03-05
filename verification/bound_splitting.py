import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# ============================================================
# 1. PENSIEVE MODEL
# ============================================================

class PensieveActor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.conv1_actor = nn.Linear(8, 128)
        self.conv2_actor = nn.Linear(8, 128)
        self.conv3_actor = nn.Linear(6, 128)

        self.fc1_actor = nn.Linear(1, 128)
        self.fc2_actor = nn.Linear(1, 128)
        self.fc3_actor = nn.Linear(1, 128)
        self.fc4_actor = nn.Linear(128 * 6, 128)

        self.pi_head = nn.Linear(128, a_dim)

    def forward(self, x):
        h1 = F.relu(self.conv1_actor(x[:, 0, :]))
        h2 = F.relu(self.conv2_actor(x[:, 1, :]))
        h3 = F.relu(self.conv2_actor(x[:, 2, :]))
        h4 = F.relu(self.conv3_actor(x[:, :, 0]))

        h5 = F.relu(self.fc1_actor(x[:, 3, 0:1]))
        h6 = F.relu(self.fc2_actor(x[:, 4, 0:1]))
        h7 = F.relu(self.fc3_actor(x[:, 5, 0:1]))

        concat = torch.cat([h1, h2, h3, h4, h5, h6], dim=1)
        hidden = F.relu(self.fc4_actor(concat))
        logits = self.pi_head(hidden)
        return logits


# ============================================================
# 2. DOMINANCE WRAPPER (y_k - y_i)
# ============================================================

class LogitDominance(nn.Module):
    def __init__(self, net, k):
        super().__init__()
        self.net = net
        self.k = k

    def forward(self, x):
        y = self.net(x)
        yk = y[:, self.k:self.k+1]
        others = torch.cat([y[:, :self.k], y[:, self.k+1:]], dim=1)
        return yk - others


# ============================================================
# 3. VERIFY REGION FOR ONE ACTION
# ============================================================

def verify_action(model, lb, ub):
    center = (lb + ub) / 2
    eps = (ub - lb) / 2

    ptb = PerturbationLpNorm(norm=float("inf"), eps=eps)
    x = BoundedTensor(center, ptb)

    bounded_model = BoundedModule(model, center, device=lb.device)

    out_lb, _ = bounded_model.compute_bounds(
        x=(x,),
        method="CROWN-Optimized"
    )

    return (out_lb >= 0).all().item()


# ============================================================
# 4. CHECK IF ANY ACTION DOMINATES
# ============================================================

def verify_any_action(net, lb, ub):
    for k in range(6):
        wrapped = LogitDominance(net, k).to(lb.device)
        if verify_action(wrapped, lb, ub):
            return True, k
    return False, None


# ============================================================
# 5. SPLIT ONLY throughput[2,7]
# ============================================================

def split_box(lb, ub):
    mid = (lb[0,2,7] + ub[0,2,7]) / 2

    lb1, ub1 = lb.clone(), ub.clone()
    ub1[0,2,7] = mid

    lb2, ub2 = lb.clone(), ub.clone()
    lb2[0,2,7] = mid

    return (lb1, ub1), (lb2, ub2)


# ============================================================
# 6. HELPERS
# ============================================================

def _make_node_info(node_id, parent_id, depth, level, lb, ub, status, action=None):
    """Build a node info dict from tensor bounds."""
    return {
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "level": level,
        "throughput_lb": lb[0,2,7].item(),
        "throughput_ub": ub[0,2,7].item(),
        "throughput_all_lb": [lb[0,2,i].item() for i in range(8)],
        "throughput_all_ub": [ub[0,2,i].item() for i in range(8)],
        "download_time_lb": [lb[0,3,i].item() for i in range(8)],
        "download_time_ub": [ub[0,3,i].item() for i in range(8)],
        "last_br_lb": lb[0,0,7].item(),
        "last_br_ub": ub[0,0,7].item(),
        "current_br_lb": lb[0,4,0].item() if lb.shape[2] > 0 else 0.0,
        "current_br_ub": ub[0,4,0].item() if ub.shape[2] > 0 else 0.0,
        "buffer_lb": lb[0,1,7].item(),
        "buffer_ub": ub[0,1,7].item(),
        "status": status,
        "action": int(action) if action is not None else None,
        "children": [],
    }


def _write_node(jsonl_file, node_info):
    """Write a single node as one JSON line and flush immediately."""
    jsonl_file.write(json.dumps(node_info) + "\n")
    jsonl_file.flush()


# ============================================================
# 7. BRANCH & BOUND
# ============================================================

def bab_search(net, init_lb, init_ub, max_depth=8, log_file=None, jsonl_file=None, level=0):
    """
    Branch and bound search.

    Only the root SPLIT node and SAFE leaf nodes are written to jsonl_file,
    one line at a time — no in-memory accumulation of node data.

    Parameters
    ----------
    net        : nn.Module
    init_lb    : Tensor  (1, 6, 8)
    init_ub    : Tensor  (1, 6, 8)
    max_depth  : int
    log_file   : file|None   — text log handle
    jsonl_file : file|None   — JSONL output handle (written incrementally)
    level      : int         — verification round number

    Returns
    -------
    safe_regions : list of (lb, ub, action)
    """

    queue = [(init_lb, init_ub, 0)]
    safe_regions = []
    node_counter = [1]

    # ── root node ─────────────────────────────────────────────────────────
    root_id = 1
    root_info = _make_node_info(root_id, None, 0, level, init_lb, init_ub, "SPLIT")
    if jsonl_file:
        _write_node(jsonl_file, root_info)

    while queue:
        lb, ub, depth = queue.pop()

        safe, action = verify_any_action(net, lb, ub)

        if safe:
            node_counter[0] += 1
            safe_info = _make_node_info(
                node_counter[0], root_id, depth, level, lb, ub, "SAFE", action
            )
            if jsonl_file:
                _write_node(jsonl_file, safe_info)
            safe_regions.append((lb.clone(), ub.clone(), action))
            continue

        if depth >= max_depth:
            # Silently discard — not logged
            continue

        (l1, u1), (l2, u2) = split_box(lb, ub)
        queue.append((l1, u1, depth + 1))
        queue.append((l2, u2, depth + 1))

    return safe_regions


# ============================================================
# 8. PUBLIC ENTRY POINT
# ============================================================

def bound_splitting(net, lb, ub, log_file=None, jsonl_file=None, level=0):
    """
    Run bound splitting and return certified safe regions.

    Nodes are written to jsonl_file immediately as they are certified —
    no node list is held in memory or returned.

    Parameters
    ----------
    net        : nn.Module
    lb         : Tensor  (1, 6, 8)
    ub         : Tensor  (1, 6, 8)
    log_file   : file|None   — text log handle
    jsonl_file : file|None   — JSONL output handle (written incrementally)
    level      : int         — verification round number

    Returns
    -------
    regions : list of (lb, ub, action)
    """
    regions = bab_search(
        net, lb, ub,
        max_depth=8,
        log_file=log_file,
        jsonl_file=jsonl_file,
        level=level,
    )

    if log_file:
        log_file.write("\n=== BOUND SPLITTING RESULTS ===\n\n")
        log_file.write(f"Safe Regions ({len(regions)} found):\n")
        for i, (lb_r, ub_r, action) in enumerate(regions, 1):
            log_file.write(f"  Region {i}: "
                           f"Throughput {lb_r[0,2,7].item():.6f}"
                           f" ~ {ub_r[0,2,7].item():.6f}"
                           f"  Action: {action}\n")
        log_file.flush()

    print("\nSAFE REGIONS FOUND:\n")
    for i, (lb_r, ub_r, action) in enumerate(regions):
        print(f"Region {i+1}: Throughput "
              f"{lb_r[0,2,7].item():.6f} ~ {ub_r[0,2,7].item():.6f}"
              f" -> Action {action}")

    return regions