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

def verify_action(lirpa_model, lb, ub):
    ptb = PerturbationLpNorm(
        norm=float("inf"),
        x_L=lb,
        x_U=ub
    )

    x = BoundedTensor((lb + ub) / 2, ptb)

    out_lb, _ = lirpa_model.compute_bounds(
        x=(x,),
        method="CROWN-Optimized"
    )

    return (out_lb >= 0).all().item()


# ============================================================
# 4. CHECK IF ANY ACTION DOMINATES
# ============================================================

def verify_any_action(dominance_models, lb, ub):
    for k in range(6):
        lirpa_model = dominance_models[k]
        if verify_action(lirpa_model, lb, ub):
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
# 7. BRANCH & BOUND  (pure compute — no logging)
# ============================================================

def bab_search_compute(lirpa_model, dominance_models, init_lb, init_ub,
                       first_node_id, root_id, max_depth=8):
    """
    Pure-compute branch-and-bound.  No file I/O — safe to run in a
    multiprocessing worker.

    Node IDs start from first_node_id + 1 (root = first_node_id + 1).
    A counter list [first_node_id] is used internally so the caller can
    reconstruct globally-unique IDs without a shared mutable counter.

    Parameters
    ----------
    lirpa_model      : BoundedModule  (worker-local)
    dominance_models : list[BoundedModule]  (worker-local)
    init_lb          : Tensor  (1, 6, 8)
    init_ub          : Tensor  (1, 6, 8)
    first_node_id    : int  — last globally-assigned ID before this call;
                             root will be first_node_id + 1
    root_id          : int  — node_id of the root of this subtree
                             (= first_node_id + 1, pre-computed by caller)
    max_depth        : int

    Returns
    -------
    safe_regions : list of (lb, ub, action, node_id)
    node_records : list of dict  — all node_info dicts, in creation order,
                                   ready to be written to JSONL by the caller
    next_node_id : int  — last ID assigned, so the main process can advance
                          its counter correctly
    """
    counter = [first_node_id]

    # root node
    counter[0] += 1
    root_info = _make_node_info(root_id, None, 0, 0,
                                init_lb, init_ub, "SPLIT")
    node_records = [root_info]

    queue = [(init_lb, init_ub, 0)]
    safe_regions = []

    while queue:
        lb, ub, depth = queue.pop()

        safe, action = verify_any_action(dominance_models, lb, ub)

        if safe:
            counter[0] += 1
            safe_info = _make_node_info(
                counter[0], root_id, depth, 0, lb, ub, "SAFE", action
            )
            node_records.append(safe_info)
            safe_regions.append((lb.clone(), ub.clone(), action, counter[0]))
            continue

        if depth >= max_depth:
            continue

        (l1, u1), (l2, u2) = split_box(lb, ub)
        queue.append((l1, u1, depth + 1))
        queue.append((l2, u2, depth + 1))

    return safe_regions, node_records, counter[0]


# ============================================================
# 8. LOG RESULTS  (main-process only — called after pool.map)
# ============================================================

def log_bab_results(safe_regions, node_records, parent_node_id, level,
                    log_file=None, jsonl_file=None):
    """
    Write tree nodes and summary to log files.  Always called in the main
    process after bab_search_compute results have been collected.

    Parameters
    ----------
    safe_regions    : list of (lb, ub, action, node_id)  — from bab_search_compute
    node_records    : list of dict  — from bab_search_compute
    parent_node_id  : int | None
    level           : int
    log_file        : file | None
    jsonl_file      : file | None
    """
    if jsonl_file:
        # Patch parent_id and level onto the root node (worker didn't know them)
        if node_records:
            node_records[0]["parent_id"] = parent_node_id
            node_records[0]["level"]     = level
        for rec in node_records:
            rec["level"] = level          # propagate level to all nodes
            _write_node(jsonl_file, rec)

    if log_file:
        log_file.write("\n=== BOUND SPLITTING RESULTS ===\n\n")
        log_file.write(f"Safe Regions ({len(safe_regions)} found):\n")
        for i, (lb_r, ub_r, action, nid) in enumerate(safe_regions, 1):
            log_file.write(f"  Region {i} [Node {nid}]: "
                           f"Throughput {lb_r[0,2,7].item():.6f}"
                           f" ~ {ub_r[0,2,7].item():.6f}"
                           f"  Action: {action}\n")
        log_file.flush()

    print("\nSAFE REGIONS FOUND:\n")
    for i, (lb_r, ub_r, action, nid) in enumerate(safe_regions):
        print(f"Region {i+1} [Node {nid}]: Throughput "
              f"{lb_r[0,2,7].item():.6f} ~ {ub_r[0,2,7].item():.6f}"
              f" -> Action {action}")


# ============================================================
# 9. BRANCH & BOUND  (original combined API — kept for compatibility)
# ============================================================

def bab_search(lirpa_model, dominance_models, init_lb, init_ub, node_counter, parent_node_id,
               max_depth=8, log_file=None, jsonl_file=None, level=0):
    """
    Original combined compute+log entry point.  Use this when calling from
    the main process directly (no multiprocessing).

    Internally delegates to bab_search_compute + log_bab_results so there
    is no code duplication.
    """
    node_counter[0] += 1
    root_id = node_counter[0]
    first_node_id = root_id - 1   # bab_search_compute will increment to root_id

    safe_regions, node_records, last_id = bab_search_compute(
        lirpa_model, dominance_models, init_lb, init_ub,
        first_node_id=first_node_id,
        root_id=root_id,
        max_depth=max_depth,
    )

    # Advance the shared counter to account for all nodes created in the worker
    node_counter[0] = last_id

    log_bab_results(safe_regions, node_records, parent_node_id, level,
                    log_file=log_file, jsonl_file=jsonl_file)

    return safe_regions


# ============================================================
# 10. PUBLIC ENTRY POINT
# ============================================================

def bound_splitting(lirpa_model, dominance_models, lb, ub, node_counter, parent_node_id=None,
                    log_file=None, jsonl_file=None, level=0):
    """
    Run bound splitting and return certified safe regions.

    Parameters
    ----------
    lirpa_model     : BoundedModule
    dominance_models: list[BoundedModule]
    lb              : Tensor  (1, 6, 8)
    ub              : Tensor  (1, 6, 8)
    node_counter    : list[int]  — shared mutable counter, e.g. [0]
    parent_node_id  : int | None — links this root to a parent safe region
    log_file        : file | None
    jsonl_file      : file | None  — JSONL output handle (written incrementally)
    level           : int

    Returns
    -------
    regions : list of (lb, ub, action, node_id)
    """
    regions = bab_search(
        lirpa_model, dominance_models, lb, ub,
        node_counter=node_counter,
        parent_node_id=parent_node_id,
        max_depth=8,
        log_file=log_file,
        jsonl_file=jsonl_file,
        level=level,
    )
    return regions