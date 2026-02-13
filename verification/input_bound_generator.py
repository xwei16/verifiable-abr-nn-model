import numpy as np
import os

# =============================================================================
# Pensieve input structure: s = np.zeros((6, 8), dtype=np.float32)
#
#   s[0, -1]  = Last chunk bitrate          → flat index 7
#   s[1, -1]  = Last buffer size            → flat index 15
#   s[2, -1]  = Last throughput             → flat index 23
#   s[3, -1]  = Last download time          → flat index 31
#   s[4, :6]  = Next chunk sizes (6 levels) → flat indices 32–37
#   s[5, -1]  = Chunks left                 → flat index 47
#
# All other indices are structurally 0 and must be pinned to [0, 0].
# =============================================================================

# Flat index map (row-major flattening of 6x8)
IDX_BITRATE      = 7         # s[0, 7]
IDX_BUFFER       = 15        # s[1, 7]
IDX_THROUGHPUT   = 23        # s[2, 7]
IDX_DOWNLOADTIME = 31        # s[3, 7]
IDX_CHUNKSIZES   = slice(32, 38)  # s[4, 0:6]
IDX_CHUNKS_LEFT  = 47        # s[5, 7]


def create_vnnlib_from_bounds(lower, upper, output_file, metadata=None):
    """Create a VNNLIB file from lower/upper bound arrays."""
    input_dim = 48
    output_dim = 6

    with open(output_file, "w") as f:
        # Declare variables
        for i in range(input_dim):
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(output_dim):
            f.write(f"(declare-const Y_{i} Real)\n")

        # Metadata comments
        if metadata:
            f.write(f"\n; Scenario: {metadata.get('name', 'Unknown')}\n")
            for key, value in metadata.items():
                if key != 'name':
                    f.write(f"; {key}: {value}\n")

        f.write("\n; Input constraints\n")
        for i in range(input_dim):
            f.write(f"(assert (>= X_{i} {lower[i]:.10f}))\n")
            f.write(f"(assert (<= X_{i} {upper[i]:.10f}))\n")

        # Unconstrained output bounds
        f.write("\n; Output constraints (unconstrained)\n")
        for i in range(output_dim):
            f.write(f"(assert (>= Y_{i} -1000000000.0))\n")
            f.write(f"(assert (<= Y_{i}  1000000000.0))\n")


def build_bounds(scenario):
    """
    Build lower/upper bound arrays that correctly reflect Pensieve's
    sparse (6, 8) input tensor when flattened to 48 elements.

    Only the slots that Pensieve actually writes are given non-zero ranges.
    Every other index is pinned to exactly 0.
    """
    lower = np.zeros(48, dtype=np.float64)
    upper = np.zeros(48, dtype=np.float64)

    # s[0, -1]: last chunk bitrate  (index 7)
    lower[IDX_BITRATE] = scenario['bitrate'][0]
    upper[IDX_BITRATE] = scenario['bitrate'][1]

    # s[1, -1]: last buffer size  (index 15)
    lower[IDX_BUFFER] = scenario['buffer'][0]
    upper[IDX_BUFFER] = scenario['buffer'][1]

    # s[2, -1]: last throughput  (index 23)
    lower[IDX_THROUGHPUT] = scenario['throughput'][0]
    upper[IDX_THROUGHPUT] = scenario['throughput'][1]

    # s[3, -1]: last download time  (index 31)
    lower[IDX_DOWNLOADTIME] = scenario['download_time'][0]
    upper[IDX_DOWNLOADTIME] = scenario['download_time'][1]

    # s[4, :6]: next chunk sizes for each quality level  (indices 32–37)
    lower[IDX_CHUNKSIZES] = scenario['chunk_sizes'][0]
    upper[IDX_CHUNKSIZES] = scenario['chunk_sizes'][1]
    # indices 38–39 (s[4, 6] and s[4, 7]) remain 0 — never written by Pensieve

    # s[5, -1]: chunks left  (index 47)
    lower[IDX_CHUNKS_LEFT] = scenario['remaining'][0]
    upper[IDX_CHUNKS_LEFT] = scenario['remaining'][1]

    return lower, upper


def create_tight_scenarios():
    """
    Generate tight VNNLIB specs with physically consistent constraints,
    using the correct Pensieve input layout.
    """

    scenarios = [
        # ===== BANDWIDTH SCENARIOS =====
        {
            'name': '1_very_high_bw_empty_buffer',
            'description': 'Excellent connection, startup phase',
            'throughput':    (4.0, 5.0),
            'download_time': (0.5, 0.8),
            'chunk_sizes':   (500, 1500),
            'buffer':        (0.0, 1.0),
            'remaining':     (40.0, 50.0),
            'bitrate':       (0.0, 1.0),
        },
        {
            'name': '2_high_bw_partial_buffer',
            'description': 'Good connection, building buffer',
            'throughput':    (3.0, 4.0),
            'download_time': (0.6, 1.0),
            'chunk_sizes':   (800, 2000),
            'buffer':        (2.0, 4.0),
            'remaining':     (20.0, 40.0),
            'bitrate':       (1.0, 3.0),
        },
        {
            'name': '3_high_bw_full_buffer',
            'description': 'Good connection, stable playback',
            'throughput':    (3.0, 4.5),
            'download_time': (0.5, 0.9),
            'chunk_sizes':   (1000, 2500),
            'buffer':        (7.0, 10.0),
            'remaining':     (5.0, 20.0),
            'bitrate':       (3.0, 5.0),
        },
        {
            'name': '4_medium_bw_empty_buffer',
            'description': 'Average connection, startup',
            'throughput':    (1.5, 2.5),
            'download_time': (1.0, 1.8),
            'chunk_sizes':   (500, 1500),
            'buffer':        (0.0, 1.0),
            'remaining':     (40.0, 50.0),
            'bitrate':       (0.0, 1.0),
        },
        {
            'name': '5_medium_bw_partial_buffer',
            'description': 'Average connection, steady state',
            'throughput':    (1.5, 2.5),
            'download_time': (1.0, 1.8),
            'chunk_sizes':   (800, 2000),
            'buffer':        (4.0, 6.0),
            'remaining':     (15.0, 35.0),
            'bitrate':       (1.0, 3.0),
        },
        {
            'name': '6_medium_bw_full_buffer',
            'description': 'Average connection, stable playback',
            'throughput':    (1.5, 2.5),
            'download_time': (1.0, 1.8),
            'chunk_sizes':   (1000, 2000),
            'buffer':        (7.0, 10.0),
            'remaining':     (5.0, 20.0),
            'bitrate':       (2.0, 4.0),
        },
        {
            'name': '7_low_bw_empty_buffer',
            'description': 'Poor connection, startup',
            'throughput':    (0.5, 1.2),
            'download_time': (2.0, 3.5),
            'chunk_sizes':   (300, 1000),
            'buffer':        (0.0, 0.5),
            'remaining':     (40.0, 50.0),
            'bitrate':       (0.0, 0.5),
        },
        {
            'name': '8_low_bw_partial_buffer',
            'description': 'Poor connection, struggling',
            'throughput':    (0.5, 1.2),
            'download_time': (2.0, 3.5),
            'chunk_sizes':   (300, 1000),
            'buffer':        (1.0, 3.0),
            'remaining':     (20.0, 40.0),
            'bitrate':       (0.0, 2.0),
        },
        {
            'name': '9_low_bw_full_buffer',
            'description': 'Poor connection, recovered buffer',
            'throughput':    (0.5, 1.2),
            'download_time': (2.0, 3.5),
            'chunk_sizes':   (300, 800),
            'buffer':        (7.0, 10.0),
            'remaining':     (10.0, 25.0),
            'bitrate':       (0.0, 2.0),
        },

        # ===== CRITICAL SCENARIOS =====
        {
            'name': '10_very_low_bw_critical',
            'description': 'Extremely poor connection, rebuffer risk',
            'throughput':    (0.3, 0.7),
            'download_time': (3.0, 4.0),
            'chunk_sizes':   (200, 600),
            'buffer':        (0.0, 1.0),
            'remaining':     (20.0, 40.0),
            'bitrate':       (0.0, 1.0),
        },
        {
            'name': '11_bandwidth_drop',
            'description': 'Sudden bandwidth degradation',
            'throughput':    (0.8, 1.5),
            'download_time': (1.8, 3.0),
            'chunk_sizes':   (800, 2000),
            'buffer':        (2.0, 4.0),
            'remaining':     (15.0, 30.0),
            'bitrate':       (2.0, 4.0),
        },
        {
            'name': '12_bandwidth_spike',
            'description': 'Sudden bandwidth improvement',
            'throughput':    (2.5, 4.0),
            'download_time': (0.7, 1.2),
            'chunk_sizes':   (500, 1200),
            'buffer':        (5.0, 8.0),
            'remaining':     (10.0, 25.0),
            'bitrate':       (0.0, 2.0),
        },

        # ===== EDGE CASES =====
        {
            'name': '13_almost_finished_high_bw',
            'description': 'End of video, good connection',
            'throughput':    (3.0, 5.0),
            'download_time': (0.5, 1.0),
            'chunk_sizes':   (800, 2000),
            'buffer':        (5.0, 10.0),
            'remaining':     (1.0, 5.0),
            'bitrate':       (2.0, 5.0),
        },
        {
            'name': '14_almost_finished_low_bw',
            'description': 'End of video, poor connection',
            'throughput':    (0.5, 1.5),
            'download_time': (1.5, 3.0),
            'chunk_sizes':   (300, 1000),
            'buffer':        (3.0, 7.0),
            'remaining':     (1.0, 5.0),
            'bitrate':       (0.0, 2.0),
        },
        {
            'name': '15_stable_excellent',
            'description': 'Ideal conditions throughout',
            'throughput':    (4.0, 5.0),
            'download_time': (0.5, 0.8),
            'chunk_sizes':   (1500, 2500),
            'buffer':        (8.0, 10.0),
            'remaining':     (10.0, 30.0),
            'bitrate':       (4.0, 5.0),
        },
    ]

    os.makedirs("spec/tight_scenarios", exist_ok=True)

    print("=" * 80)
    print("GENERATING TIGHT SCENARIO SPECIFICATIONS")
    print("Input layout: Pensieve (6x8) flattened, sparse — only live slots bounded")
    print("=" * 80)

    for scenario in scenarios:
        lower, upper = build_bounds(scenario)

        output_file = f"spec/tight_scenarios/{scenario['name']}.vnnlib"

        metadata = {
            'name': scenario['name'],
            'description': scenario['description'],
            'throughput_range':    f"{scenario['throughput'][0]}-{scenario['throughput'][1]} Mbps  → X_23",
            'download_time_range': f"{scenario['download_time'][0]}-{scenario['download_time'][1]} sec  → X_31",
            'buffer_range':        f"{scenario['buffer'][0]}-{scenario['buffer'][1]} sec  → X_15",
            'bitrate_range':       f"{scenario['bitrate'][0]}-{scenario['bitrate'][1]}  → X_7",
            'chunk_sizes_range':   f"{scenario['chunk_sizes'][0]}-{scenario['chunk_sizes'][1]} KB  → X_32..X_37",
            'remaining_range':     f"{scenario['remaining'][0]}-{scenario['remaining'][1]}  → X_47",
            'zero_indices':        'all others pinned to 0',
        }

        create_vnnlib_from_bounds(lower, upper, output_file, metadata)

        bw_range  = scenario['throughput'][1]   - scenario['throughput'][0]
        buf_range = scenario['buffer'][1]        - scenario['buffer'][0]

        print(f"\n{scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"  Throughput:    {scenario['throughput'][0]:.1f}–{scenario['throughput'][1]:.1f} Mbps (Δ={bw_range:.1f})  X_23")
        print(f"  Download time: {scenario['download_time'][0]:.1f}–{scenario['download_time'][1]:.1f} sec                X_31")
        print(f"  Buffer:        {scenario['buffer'][0]:.1f}–{scenario['buffer'][1]:.1f} sec (Δ={buf_range:.1f})       X_15")
        print(f"  Bitrate:       {scenario['bitrate'][0]:.1f}–{scenario['bitrate'][1]:.1f}                        X_7")
        print(f"  Chunk sizes:   {scenario['chunk_sizes'][0]}–{scenario['chunk_sizes'][1]} KB                  X_32..X_37")
        print(f"  Chunks left:   {scenario['remaining'][0]:.1f}–{scenario['remaining'][1]:.1f}                       X_47")

    print("\n" + "=" * 80)
    print(f"✅ Generated {len(scenarios)} tight scenario specifications")
    print(f"   Output directory: spec/tight_scenarios/")
    print("=" * 80)

    return scenarios


def create_ultra_tight_scenarios():
    """
    Point-wise specs with ±5% epsilon balls around realistic operating points.
    """

    operating_points = [
        {
            'name': 'point_1_high_bw_full_buffer',
            'center': {
                'throughput':    4.0,
                'download_time': 0.7,
                'chunk_sizes':   1800,
                'buffer':        8.5,
                'remaining':     15.0,
                'bitrate':       4.0,
            },
            'epsilon': 0.05,
        },
        {
            'name': 'point_2_low_bw_empty_buffer',
            'center': {
                'throughput':    0.8,
                'download_time': 2.8,
                'chunk_sizes':   600,
                'buffer':        0.5,
                'remaining':     45.0,
                'bitrate':       0.5,
            },
            'epsilon': 0.05,
        },
        {
            'name': 'point_3_medium_bw_medium_buffer',
            'center': {
                'throughput':    2.0,
                'download_time': 1.4,
                'chunk_sizes':   1200,
                'buffer':        5.0,
                'remaining':     25.0,
                'bitrate':       2.5,
            },
            'epsilon': 0.05,
        },
    ]

    os.makedirs("spec/ultra_tight", exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING ULTRA-TIGHT SPECIFICATIONS (±5% epsilon)")
    print("Input layout: Pensieve (6x8) flattened, sparse — only live slots bounded")
    print("=" * 80)

    for point in operating_points:
        lower = np.zeros(48, dtype=np.float64)
        upper = np.zeros(48, dtype=np.float64)
        c   = point['center']
        eps = point['epsilon']

        # s[0, -1]: bitrate  → index 7
        lower[IDX_BITRATE]      = max(0.0, c['bitrate']      * (1 - eps))
        upper[IDX_BITRATE]      =          c['bitrate']      * (1 + eps)

        # s[1, -1]: buffer   → index 15
        lower[IDX_BUFFER]       = max(0.0, c['buffer']       * (1 - eps))
        upper[IDX_BUFFER]       =          c['buffer']       * (1 + eps)

        # s[2, -1]: throughput → index 23
        lower[IDX_THROUGHPUT]   =          c['throughput']   * (1 - eps)
        upper[IDX_THROUGHPUT]   =          c['throughput']   * (1 + eps)

        # s[3, -1]: download time → index 31
        lower[IDX_DOWNLOADTIME] =          c['download_time']* (1 - eps)
        upper[IDX_DOWNLOADTIME] =          c['download_time']* (1 + eps)

        # s[4, :6]: chunk sizes → indices 32–37
        lower[IDX_CHUNKSIZES]   =          c['chunk_sizes']  * (1 - eps)
        upper[IDX_CHUNKSIZES]   =          c['chunk_sizes']  * (1 + eps)

        # s[5, -1]: chunks left → index 47
        lower[IDX_CHUNKS_LEFT]  =          c['remaining']    * (1 - eps)
        upper[IDX_CHUNKS_LEFT]  =          c['remaining']    * (1 + eps)

        output_file = f"spec/ultra_tight/{point['name']}.vnnlib"

        metadata = {
            'name':              point['name'],
            'epsilon':           f"±{eps*100:.0f}%",
            'center_throughput': f"{c['throughput']:.2f} Mbps  → X_23",
            'center_buffer':     f"{c['buffer']:.2f} sec       → X_15",
            'center_bitrate':    f"{c['bitrate']:.2f}           → X_7",
            'center_chunks':     f"{c['chunk_sizes']:.0f} KB    → X_32..X_37",
            'center_remaining':  f"{c['remaining']:.1f}         → X_47",
            'zero_indices':      'all others pinned to 0',
        }

        create_vnnlib_from_bounds(lower, upper, output_file, metadata)

        print(f"\n{point['name']}")
        print(f"  Epsilon: ±{eps*100:.0f}%")
        print(f"  BW={c['throughput']:.2f} Mbps (X_23), Buf={c['buffer']:.2f} sec (X_15), "
              f"Bitrate={c['bitrate']:.2f} (X_7), Chunks={c['chunk_sizes']:.0f} KB (X_32..37)")

    print("\n" + "=" * 80)
    print(f"✅ Generated {len(operating_points)} ultra-tight specifications")
    print("=" * 80)


if __name__ == "__main__":
    create_tight_scenarios()
    create_ultra_tight_scenarios()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Run verification on tight scenarios:")
    print("   python batch_verify.py --spec-dir spec/tight_scenarios/")
    print("\n2. If bounds still overlap, try ultra-tight:")
    print("   python batch_verify.py --spec-dir spec/ultra_tight/")
    print("\n3. Analyse results:")
    print("   python analyze_all_results.py")
    print("=" * 80)