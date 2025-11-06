#!/usr/bin/env python3
# pgd_one_attack.py  ----------------------------------------------
"""
Targeted PGD against Pensieve's actor network - PyTorch implementation
  • pushes every state toward class-0 (300 kbps)
  • perturbs only one feature (Last1_downloadtime) and respects both an L∞ budget (eps)
    and the physical bounds in qoe_spec.json
Outputs a new CSV "pgd_attacked_data.csv" whose feature has been
adversarially modified; all other columns stay identical.
"""

import numpy as np
import torch
import os
import numpy as np
import ppo2 as network


'''
FEATURES = [                         # feature(s) to attack
    "Last1_downloadtime"
]

BRS = [300, 750, 1200, 1850, 2850, 4300]

ALL_FEATURES = [
    'Last1_chunk_bitrate', 'Last1_buffer_size', 'Last8_throughput', 'Last7_throughput',
    'Last6_throughput', 'Last5_throughput', 'Last4_throughput', 'Last3_throughput',
    'Last2_throughput', 'Last1_throughput', 'Last8_downloadtime', 'Last7_downloadtime',
    'Last6_downloadtime', 'Last5_downloadtime', 'Last4_downloadtime', 'Last3_downloadtime',
    'Last2_downloadtime', 'Last1_downloadtime', 'chunksize1', 'chunksize2', 'chunksize3',
    'chunksize4', 'chunksize5', 'chunksize6', 'Chunks_left', 
    "br","qoe_2"
]
'''


class PGD:

    def __init__(self):
        MODEL_PATH   = "../pensieve_rl_model/nn_model_ep_155400.pth"

        # PPO2 parameters
        S_DIM = [6, 8]
        A_DIM = 6
        ACTOR_LR_RATE = 1e-4

        self.model = self.load_ppo2_model(MODEL_PATH, S_DIM, A_DIM, ACTOR_LR_RATE)
        self.target_class = 0                    # target class [0: 300 kbps, 1: 1000 kbps]
        self.device = torch.device("cpu")
        self.attack_idxs = [31]
        self.steps = 40                          # number of PGD steps
        self.alpha = 0.001                       # step size
        self.eps   = 0.01                        # epsilon for L∞ bound


    def load_ppo2_model(self, model_path, state_dim, action_dim, learning_rate):
        if os.path.exists(model_path):
            actor = network.Network(state_dim=state_dim,
                                    action_dim=action_dim,
                                    learning_rate=learning_rate)
            model = actor.load_model(model_path)
            model = actor
            # print(model)

            return model
        else:
            raise FileNotFoundError(f"PyTorch model not found at {model_path}")
    
    def attack_download(self, history_state):

        # return downloading time after attack

        state0 = history_state
        x0 = state0[3, 7].astype(np.float32)

        state0 = np.expand_dims(state0, axis=0)
        pred = self.model.predict(state0)
        pred = np.argmax(pred)
        
        if pred == self.target_class:
            return x0
        
        x_adv = x0
        x_adv_prev = x0
        state_cur = None
        thrpt_times = None

        # we target 10% of purtabation
        low_bound = x0 * 0.9
        high_bound = x0 * 1.1

        
        for _ in range(self.steps):
            state_cur = state0.flatten()
            state_cur[self.attack_idxs[0]] = float(x_adv)
            thrpt_times = x_adv_prev / x_adv
            state_cur[self.attack_idxs[0] - 8] *= thrpt_times
            state_cur = state_cur.reshape(state0.shape)

            state_cur = torch.from_numpy(state_cur).to(torch.float32).requires_grad_(True)

            output = self.model.forward(state_cur)
            # Add batch dimension
            if output.dim() == 1:
                output = output.unsqueeze(0)    # Now shape: (1, 6)
            target = torch.tensor([self.target_class], device=self.device)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            grad = state_cur.grad.view(-1)[self.attack_idxs[0]]

            x_perturbation = self.alpha * torch.sign(grad)
            x_adv_prev = x_adv
            x_adv -= x_perturbation
            x_adv = torch.clamp(x_adv, x0 - self.eps, x0 + self.eps)
            x_adv = torch.clamp(x_adv, low_bound, high_bound)
        
        return x_adv


    ## Verify attack effectiveness
    def attack_evaluation(self, state_orig, x_adv, x_orig):
        # x: downloading time
        state_orig_flat = state_orig.flatten()
        state_adv = state_orig_flat.copy()
        thrpt_times = x_orig / x_adv
        state_adv[self.attack_idxs[0]] =x_adv
        state_adv[self.attack_idxs[0] - 8] *= thrpt_times
        state_adv = state_adv.reshape(state_orig.shape)

        state_orig = np.expand_dims(state_orig, axis=0)
        state_adv = np.expand_dims(state_adv, axis=0)

        p_orig = np.argmax(self.model.predict(state_orig))
        p_adv = np.argmax(self.model.predict(state_adv))

        if p_adv < p_orig or p_adv == 0:
            return True # success
        else:
            return False
