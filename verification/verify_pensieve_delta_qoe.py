# generate_delta_qoe_bounds.py
"""
Generate delta QoE bounds for Pensieve ABR.

delta_qoe(action) =
    q(bitrate_current) - |q(bitrate_current) - q(bitrate_previous)|

where:
    q(bitrate) = log(bitrate_kbps / MIN_BITRATE_KBPS)
"""

import numpy as np
import argparse
from typing import Tuple


class PensieveDeltaQoEGenerator:
    """
    Compute bounds on:

        delta_qoe = q(current) - |q(current) - q(previous)|

    for each bitrate action.
    """

    BITRATE_LEVELS = np.array([0.3, 0.75, 1.2, 1.85, 2.85, 4.3])  # Mbps
    MIN_BITRATE_KBPS = 300
    NUM_ACTIONS = 6

    def __init__(self,
                 prev_bitrate_lower: float,
                 prev_bitrate_upper: float):

        assert prev_bitrate_lower <= prev_bitrate_upper

        self.prev_bitrate_lower = prev_bitrate_lower
        self.prev_bitrate_upper = prev_bitrate_upper

    # ---------------------------------------------------------
    # Quality function
    # ---------------------------------------------------------
    def q(self, bitrate_mbps: float) -> float:
        bitrate_kbps = bitrate_mbps * 1000
        return np.log(bitrate_kbps / self.MIN_BITRATE_KBPS)

    # ---------------------------------------------------------
    # Smoothness penalty bounds
    # ---------------------------------------------------------
    def smoothness_bounds(self, current_bitrate: float) -> Tuple[float, float]:

        q_current = self.q(current_bitrate)
        q_prev_lower = self.q(self.prev_bitrate_lower)
        q_prev_upper = self.q(self.prev_bitrate_upper)

        diff_lower = q_current - q_prev_upper
        diff_upper = q_current - q_prev_lower

        # Absolute value bounds
        if diff_lower >= 0:
            penalty_lower = diff_lower
            penalty_upper = diff_upper

        elif diff_upper <= 0:
            penalty_lower = -diff_upper
            penalty_upper = -diff_lower

        else:
            penalty_lower = 0.0
            penalty_upper = max(abs(diff_lower), abs(diff_upper))

        return penalty_lower, penalty_upper

    # ---------------------------------------------------------
    # delta_qoe bounds per action
    # ---------------------------------------------------------
    def compute_delta_qoe_bounds(self, action: int) -> Tuple[float, float]:

        current_bitrate = self.BITRATE_LEVELS[action]
        q_current = self.q(current_bitrate)

        penalty_lower, penalty_upper = self.smoothness_bounds(current_bitrate)

        # delta_qoe = q_current - penalty
        delta_lower = q_current - penalty_upper
        delta_upper = q_current - penalty_lower

        return delta_lower, delta_upper

    # ---------------------------------------------------------
    # Generate all bounds
    # ---------------------------------------------------------
    def generate_bounds(self) -> dict:

        delta_lower = np.zeros(self.NUM_ACTIONS)
        delta_upper = np.zeros(self.NUM_ACTIONS)
        penalties_lower = np.zeros(self.NUM_ACTIONS)
        penalties_upper = np.zeros(self.NUM_ACTIONS)

        for i in range(self.NUM_ACTIONS):
            delta_lower[i], delta_upper[i] = self.compute_delta_qoe_bounds(i)
            penalties_lower[i], penalties_upper[i] = \
                self.smoothness_bounds(self.BITRATE_LEVELS[i])

        return {
            "delta_qoe_lower": delta_lower,
            "delta_qoe_upper": delta_upper,
            "smoothness_penalty_lower": penalties_lower,
            "smoothness_penalty_upper": penalties_upper,
            "bitrate_levels": self.BITRATE_LEVELS,
            "prev_bitrate_lower": self.prev_bitrate_lower,
            "prev_bitrate_upper": self.prev_bitrate_upper
        }


# -------------------------------------------------------------
# Loading
# -------------------------------------------------------------
def load_verification_results(npz_file: str):

    data = np.load(npz_file)

    try:
        prev_bitrate_lower = float(data['prev_bitrate_lower'])
        prev_bitrate_upper = float(data['prev_bitrate_upper'])
    except KeyError:
        print("No previous bitrate bounds found. Using [0.3, 4.3]")
        prev_bitrate_lower = 0.3
        prev_bitrate_upper = 4.3

    return prev_bitrate_lower, prev_bitrate_upper


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def generate_delta_qoe_bounds(input_file: str, output_file: str = None):

    print(f"Loading: {input_file}")
    prev_lb, prev_ub = load_verification_results(input_file)

    generator = PensieveDeltaQoEGenerator(prev_lb, prev_ub)
    bounds = generator.generate_bounds()

    if output_file is None:
        output_file = input_file.replace(".npz", "_delta_qoe.npz")

    np.savez_compressed(output_file, **bounds)
    print(f"Saved delta QoE bounds to: {output_file}")

    return bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bounds', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    generate_delta_qoe_bounds(args.bounds, args.output)


if __name__ == "__main__":
    main()