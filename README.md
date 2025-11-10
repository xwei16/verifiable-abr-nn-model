# Verifiable ML-based ABR Controllers for Network Systems

This repository provides an experimental framework for **verifying and attacking machine learning–based Adaptive Bitrate (ABR) algorithms**, particularly those derived from the **Pensieve** reinforcement learning model. The project integrates model simulation, adversarial perturbation (PGD-based), and network trace emulation for robust analysis of ABR decision behavior.

---

## 📁 Project Structure
abr-verification-reorganizing/
│
├── model/
│   └── abr-model/
│       ├── pensieve_bb_model/          # Black-box Pensieve model implementation
│       │   └── bb.py
│       └── pensieve_rl_model/          # RL-based Pensieve model weights and checkpoints
│           ├── nn_model_ep_155400.pth
│           └── pretrain_linear_reward.ckpt*
│
├── multi-step-attack/
│   ├── sabre/                          # Core simulation and attack modules
│   │   ├── sabre_new.py                # Main SABRE (Simulator for ABR Evaluation) class
│   │   ├── bb_pos.py                   # Environment interface for baseline models
│   │   ├── pensieve_pos.py             # Environment interface for Pensieve (RL) models
│   │   ├── results/                    # Attack outcome logs (e.g., successful perturbations)
│   │   ├── traces/                     # Network trace examples (e.g., abr_test/)
│   │   └── unused/                     # Legacy or experimental modules
│   └── logs/                           # Experiment log files (.csv)
│
├── .gitignore
├── .gitattributes
└── README.md                           # (This file)

## 2. Running a Simulation

```bash
cd multi-step-attack/sabre
python sabre_new.py
```

## 3. Running an Attack
```bash
cd multi-step-attack/sabre
python bb_pos.py         # or pensieve_pos.py for RL models
```

