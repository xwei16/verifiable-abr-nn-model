import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from verification_loop import load_initial_state

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

    # All dominance outputs must be >= 0
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
# 6. BRANCH & BOUND
# ============================================================

def bab_search(net, init_lb, init_ub, max_depth=8):

    queue = [(init_lb, init_ub, 0)]
    safe_regions = []

    while queue:
        lb, ub, depth = queue.pop()

        print("depth:", depth,
              "throughput range:",
              lb[0,2,7].item(), "~", ub[0,2,7].item())

        safe, action = verify_any_action(net, lb, ub)

        if safe:
            print("CERTIFIED REGION:",
                  lb[0,2,7].item(), "~", ub[0,2,7].item(),
                  "-> action", action)
            safe_regions.append((lb.clone(), ub.clone(), action))
            continue

        if depth >= max_depth:
            continue

        (l1, u1), (l2, u2) = split_box(lb, ub)

        queue.append((l1, u1, depth+1))
        queue.append((l2, u2, depth+1))

    return safe_regions


# ============================================================
# 7. RUN
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

net = PensieveActor(48, 6).to(device)

ckpt = torch.load(
    "../model/abr-model/pensieve_rl_model/nn_model_ep_155400.pth",
    map_location="cpu"
)

actor_state_dict = ckpt[0]
net.load_state_dict(actor_state_dict)

net.eval()


# Initial region
# lb = torch.zeros(1,6,8).to(device)
# ub = torch.zeros(1,6,8).to(device)
lb, ub = load_initial_state(device)


regions = bab_search(net, lb, ub, max_depth=8)


print("\nSAFE REGIONS FOUND:\n")
for r in regions:
    print("Throughput:",
          r[0][0,2,7].item(), "~",
          r[1][0,2,7].item(),
          "-> action", r[2])