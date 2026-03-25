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
# 2. DOMINANCE WRAPPER — single matrix, one bound propagation
# ============================================================

class AllDominance(nn.Module):
    """
    Replaces 6 × LogitDominance with one fixed linear layer.

    For a_dim=6 actions, builds an (a_dim*(a_dim-1), a_dim) = (30, 6) matrix M
    where block k  (rows k*5 : (k+1)*5)  encodes  y_k - y_i  for all i ≠ k.

    One forward pass → 30-dim output.
    LiRPA propagates bounds through the actor AND this layer in a single pass,
    so verify_any_action needs only one compute_bounds call instead of six.
    """
    def __init__(self, net, a_dim=6):
        super().__init__()
        self.net   = net
        self.a_dim = a_dim
        self.n_pairs = a_dim * (a_dim - 1)   # 30

        rows = []
        for k in range(a_dim):
            for i in range(a_dim):
                if i == k:
                    continue
                row = [0.0] * a_dim
                row[k] =  1.0
                row[i] = -1.0
                rows.append(row)

        # Fixed (non-trainable) weight matrix — shape (30, 6)
        M = torch.tensor(rows, dtype=torch.float32)
        self.dom = nn.Linear(a_dim, self.n_pairs, bias=False)
        self.dom.weight = nn.Parameter(M, requires_grad=False)

    def forward(self, x):
        y = self.net(x)          # (batch, 6)
        return self.dom(y)       # (batch, 30)


# ============================================================
# 3. VERIFY — single bound propagation covers all 6 actions
# ============================================================

BATCH_SIZE = 256   # number of queue nodes processed per compute_bounds call


def verify_any_action(lirpa_all_dominance, lb, ub):
    """
    Batched compute_bounds call.

    lb, ub : (B, 6, 8) tensors — B regions stacked along dim 0.
    Returns a list of B (safe: bool, action: int|None) tuples.
    """
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=lb, x_U=ub)
    x   = BoundedTensor((lb + ub) / 2, ptb)

    out_lb, _ = lirpa_all_dominance.compute_bounds(
        x=(x,), method="CROWN-Optimized"
    )                                   # (B, 30)

    a_dim = 6
    block = a_dim - 1                   # 5 comparisons per action

    results = []
    for b in range(out_lb.shape[0]):
        row = out_lb[b]                 # (30,)
        found = False
        for k in range(a_dim):
            if (row[k * block : (k + 1) * block] >= 0).all():
                results.append((True, k))
                found = True
                break
        if not found:
            results.append((False, None))

    return results


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

def bab_search_compute(lirpa_model, lirpa_all_dominance, init_lb, init_ub,
                       max_depth=8):
    """
    Pure-compute branch-and-bound.  No file I/O — safe to run in a
    worker thread.

    Node IDs are local, starting from 1 (root = 1).  The caller is
    responsible for applying a global offset via the renumber pass in
    the main thread.

    Parameters
    ----------
    lirpa_model          : BoundedModule  (worker-local)
    lirpa_all_dominance  : BoundedModule  (worker-local)
    init_lb              : Tensor  (1, 6, 8)
    init_ub              : Tensor  (1, 6, 8)
    max_depth            : int

    Returns
    -------
    safe_regions : list of (lb, ub, action, local_node_id)
    node_records : list of dict  — all node_info dicts, in creation order,
                                   with local IDs; caller applies offset
    n_local      : int  — total nodes created (= highest local ID assigned)
    """
    counter = [0]

    # root node — always local ID 1
    counter[0] += 1
    root_info = _make_node_info(1, None, 0, 0, init_lb, init_ub, "SPLIT")
    node_records = [root_info]

    queue = [(init_lb, init_ub, 0)]
    safe_regions = []

    while queue:
        # Pop up to BATCH_SIZE nodes and verify them in one compute_bounds call
        batch = []
        while queue and len(batch) < BATCH_SIZE:
            batch.append(queue.pop())

        lbs = torch.cat([lb for lb, ub, depth in batch], dim=0)   # (B, 6, 8)
        ubs = torch.cat([ub for lb, ub, depth in batch], dim=0)   # (B, 6, 8)
        verify_results = verify_any_action(lirpa_all_dominance, lbs, ubs)

        for (lb, ub, depth), (safe, action) in zip(batch, verify_results):
            if safe:
                counter[0] += 1
                safe_info = _make_node_info(
                    counter[0], 1, depth, 0, lb, ub, "SAFE", action
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
    process after bab_search_compute results have been collected and
    renumbered.

    Parameters
    ----------
    safe_regions    : list of (lb, ub, action, node_id)  — already renumbered
    node_records    : list of dict  — already renumbered
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

def bab_search(lirpa_model, lirpa_all_dominance, init_lb, init_ub, node_counter, parent_node_id,
               max_depth=8, log_file=None, jsonl_file=None, level=0):
    """
    Original combined compute+log entry point.  Use this when calling from
    the main process directly (no multiprocessing).
    """
    safe_regions, node_records, n_local = bab_search_compute(
        lirpa_model, lirpa_all_dominance, init_lb, init_ub,
        max_depth=max_depth,
    )

    # Renumber local IDs to global IDs
    off = node_counter[0]
    for rec in node_records:
        rec["node_id"] += off
        if rec["parent_id"] is not None:
            rec["parent_id"] += off
    safe_regions = [(lb, ub, a, nid + off) for lb, ub, a, nid in safe_regions]
    node_counter[0] += n_local

    log_bab_results(safe_regions, node_records, parent_node_id, level,
                    log_file=log_file, jsonl_file=jsonl_file)

    return safe_regions


# ============================================================
# 10. PUBLIC ENTRY POINT
# ============================================================

def bound_splitting(lirpa_model, lirpa_all_dominance, lb, ub, node_counter, parent_node_id=None,
                    log_file=None, jsonl_file=None, level=0):
    """
    Run bound splitting and return certified safe regions.

    Parameters
    ----------
    lirpa_model          : BoundedModule  (pensieve actor — used for logit logging only)
    lirpa_all_dominance  : BoundedModule  (AllDominance — single model for all 6 checks)
    lb                   : Tensor  (1, 6, 8)
    ub                   : Tensor  (1, 6, 8)
    node_counter         : list[int]
    parent_node_id       : int | None
    log_file             : file | None
    jsonl_file           : file | None
    level                : int

    Returns
    -------
    regions : list of (lb, ub, action, node_id)
    """
    regions = bab_search(
        lirpa_model, lirpa_all_dominance, lb, ub,
        node_counter=node_counter,
        parent_node_id=parent_node_id,
        max_depth=8,
        log_file=log_file,
        jsonl_file=jsonl_file,
        level=level,
    )
    return regions