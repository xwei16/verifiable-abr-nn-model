# Verifiable ML-based ABR Controllers for Network Systems

This repository provides a comprehensive framework for developing, analyzing, and verifying machine learning-based Adaptive Bitrate (ABR) controllers in network systems. It includes tools for adversarial testing and QoE (Quality of Experience) evaluation, trianed neural network model for network prediction, along with datasets and models for experimentation.

---

## Repository Structure

### **Data**
- **`abr-specifications/`**: Contains JSON files defining QoE specifications and segment sizes.
  - `full_spec.json`: Full QoE specification.
  - `qoe_spec.json`: QoE specification used in experiments.
  - `useful_results_from_spectra/`: Results derived from Spectra data.
- **`pensieve/`**: Data related to Pensieve ABR algorithms.
  - `filtered_good_testing_data.csv`: Filtered testing data.
  - `pgd_attacked_data.csv`: Data generated from PGD attacks.
- **`puffer/`**: Data from the Puffer streaming platform.
  - `puffer_data_cleaned/`: Cleaned Puffer data.
  - `puffer_data_original/`: Original Puffer data.
- **`spectra/`**: Contains datasets for training and testing.
  - `aurora_df_*`: Aurora datasets for various configurations.
  - `bb_events_pensieve_train.csv`: Training data for Pensieve BB events.
  - `mpc_events_pensieve_train.csv`: Training data for Pensieve MPC events.

### **Model**
- **`abr-model/`**: Models for ABR algorithms.
  - `pensieve_bb_model/`: Pensieve BB model.
  - `pensieve_rl_model/`: Pensieve RL model.
- **`network-prediction-model/`**: Models for predicting network behavior

### **Multi-Step Attack**
- **`logs/`**: Logs from multi-step adversarial attacks.
  - Example: `bb_log_2025-08-28_04-10-50.csv`
- **`sabre/`**: A simulator predicting video downloading time in every step of the attack.
- **`src/`**: Source code for multi-step adversarial attacks.

### **Network Prediction**
- **`output-models/`**: Trained models for network prediction.
- **`scripts/`**: Scripts for data processing and evaluation.
- **`src/`**: Source code for network prediction.
- **`utils/`**: Utility scripts for network prediction.

### **Single-Step Attack**
- **`results/`**: Results from single-step adversarial attacks.
- **`src/`**: Source code for single-step attacks.

### **Violation Sampling**
- **`logs/`**: Logs from violation sampling experiments.
- **`src/`**: Source code for violation sampling.
- **`utils/`**: Utility scripts for violation sampling.

---

## Key Features
- **Adversarial Attacks**: Implementations of single-step and multi-step attacks on ABR algorithms.
- **Network Prediction**: Trained Nueral Network Models for predicting network behavior, specifically video chunk downloading time, under various network conditions.
- **QoE Specifications**: Tools for evaluating ABR algorithms based on QoE metrics.
- **Data Processing**: Scripts for cleaning and preparing datasets for experiments.

---

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Statsmodels
- ...

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/xwei16/verifiable-abr-nn-model
   cd verifiable-abr-nn-model
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
- **Run Adversarial Attacks**:
  ```bash
  python3 single-step/attack/src/pgd_torch/pgd_rand_attack.py
  ```
- **Train Network Prediction Models**:
  ```bash
  python3 network-prediction/src/model_training.py
  ```
- **Evaluate Models**:
  ```bash
  python3 network-prediction/src/model_evaluation.py
  ```

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## Acknowledgments
- **[Pensieve](https://github.com/hongzimao/pensieve)**: For the base RL-based ABR model.
- **[Puffer](https://puffer.stanford.edu/)**: For the datasets used in training and evaluation.
- **[Spectra](https://arxiv.org/abs/2412.03028)**: For specification generation tool used in attacks.
- **[SABRE](https://github.com/sabre-team/sabre)**: For the simulator used to predict video downloading time during adversarial attacks.