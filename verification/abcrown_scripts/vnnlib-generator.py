import numpy as np

def spec_to_vnnlib(output_path: str):
    """
    Convert one ABR JSON spec entry into a VNNLIB file
    for a Pensieve input of shape (1, 6, 8).

    The input is flattened row-major:
        flat_index = row * 8 + col

    Parameters
    ----------
    spec : dict
        One spec entry (parsed JSON).
    output_path : str
        Path to write the .vnnlib file.
    """

    lb = np.zeros((6, 8), dtype=np.float32)
    ub = np.zeros((6, 8), dtype=np.float32)

    # ------------------------------------------------------------------
    # Row 0: last chunk bitrate (col 7)
    # ------------------------------------------------------------------
    lb[0, 7] = 0.66279
    ub[0, 7] = 0.66279

    # ------------------------------------------------------------------
    # Row 1: buffer size (col 7)
    # ------------------------------------------------------------------
    lb[1, 7] = 0.4
    ub[1, 7] = 0.5


    # ------------------------------------------------------------------
    # Row 2: throughput history — Last8 (col 0) → Last1 (col 7)
    # ------------------------------------------------------------------
    # for n in range(1, 9):
    #     col = 8 - n
    #     lb[2, col] = spec[f"Last{n}_throughput_l"]
    #     ub[2, col] = spec[f"Last{n}_throughput_u"]
    
    lb[2, 7] = 0.07
    ub[2, 7] = 0.78
    

    # ------------------------------------------------------------------
    # Row 3: download time history
    # ------------------------------------------------------------------
    # for n in range(1, 9):
    #     col = 8 - n
    #     lb[3, col] = spec[f"Last{n}_downloadtime_l"]
    #     ub[3, col] = spec[f"Last{n}_downloadtime_u"]
    
    lb[3, 7] = 0.15
    ub[3, 7] = 0.66

    # ------------------------------------------------------------------
    # Row 4: chunk sizes (cols 0..5)
    # ------------------------------------------------------------------
    chunk_size_lb = [0.11, 0.26, 0.39, 0.60, 0.89, 1.43]
    chunk_size_ub = [0.18, 0.45, 0.71, 1.08, 1.73, 2.40]
    for i in range(6):
        lb[4, i] = chunk_size_lb[i]
        ub[4, i] = chunk_size_ub[i]
    
    # ------------------------------------------------------------------
    # Row 5: chunks left (col 7)
    # ------------------------------------------------------------------
    lb[5, 7] = 0
    ub[5, 7] = 0.96

    if np.any(lb > ub):
        raise ValueError("Spec contains lb > ub for at least one input dimension.")

    lb_flat = lb.reshape(-1)
    ub_flat = ub.reshape(-1)

    with open(output_path, "w") as f:
        # --------------------------------------------------------------
        # Declare 48 input variables
        # --------------------------------------------------------------
        for i in range(48):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write("\n")

        for i in range(6):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n")


        # --------------------------------------------------------------
        # Input bounds
        # --------------------------------------------------------------
        for i in range(48):
            f.write(f"(assert (>= X_{i} {lb_flat[i]}))\n")
            f.write(f"(assert (<= X_{i} {ub_flat[i]}))\n")

        # --------------------------------------------------------------
        # Placeholder property (replace with your real spec)
        # Example: always true
        # --------------------------------------------------------------
        f.write("\n")
        # f.write("(assert true)\n")

        # Output bounds
        for i in range(1, 6):
            f.write(f"(assert (>= Y_0 Y_{i}))\n")

spec_to_vnnlib("spec/1.vnnlib")