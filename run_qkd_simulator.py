"""
Run script for Satellite-HAP-GS QKD experiments.

This follows the style of run_simulator_2.py:
- define parameters at the top
- run simulator
- export results
"""

import os
from datetime import datetime

from qkd_simulator import QKDSimulator


# =========================
# Simulation Parameters
# =========================

DATA_DIR = "data/qkd"
RESULT_DIR = "results/qkd"

NODES_FILE = os.path.join(DATA_DIR, "nodes.csv")
CAPACITIES_FILE = os.path.join(DATA_DIR, "link_capacities.csv")
DEMANDS_FILE = os.path.join(DATA_DIR, "demands.csv")

BETA_QKP = 1e-4
TIME_LIMIT = 300


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    simulator = QKDSimulator(
        nodes_file=NODES_FILE,
        capacities_file=CAPACITIES_FILE,
        demands_file=DEMANDS_FILE,
    )

    simulator.show_topology()

    simulator.run_milp(
        scenario_name="satellite_hap_gs_integrated",
        beta=BETA_QKP,
        time_limit=TIME_LIMIT,
    )

    simulator.summary()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULT_DIR, f"qkd_results_{timestamp}.csv")
    simulator.metrics.export_csv(output_file)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
