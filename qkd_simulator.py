"""
Main simulator for Satellite-HAP-GS QKD experiments.

This follows the EventSimulator pattern in the existing repository:
- initialize network state
- run a selected model
- record and summarize results
"""

from qkd_input_loader import QKDInputLoader
from qkd_network import QKDNetwork
from qkd_milp_model import QKDMILPModel
from qkd_metrics import QKDMetrics


class QKDSimulator:
    def __init__(self, nodes_file, capacities_file, demands_file):
        self.loader = QKDInputLoader(nodes_file, capacities_file, demands_file)

        self.topology = self.loader.build_topology()
        self.capacities = self.loader.load_link_capacities()
        self.demands = self.loader.load_demands()

        self.network = QKDNetwork(
            topology=self.topology,
            link_capacities=self.capacities,
        )

        self.metrics = QKDMetrics()

    def run_milp(self, scenario_name="satellite_hap_gs", beta=1e-4, time_limit=None):
        print("\n" + "#" * 80)
        print(f"### Running QKD MILP Scenario: {scenario_name}")
        print("#" * 80)

        model = QKDMILPModel(
            qkd_network=self.network,
            demands=self.demands,
            beta=beta,
            model_name=scenario_name,
        )

        solution = model.solve(msg=True, time_limit=time_limit)

        print(f"[MILP] Status: {solution['status']}")
        print(f"[MILP] Total served keys: {solution['total_served_keys']}")
        print(f"[MILP] Objective: {solution['objective']}")
        print(f"[MILP] Activated links: {solution['activated_links']}")

        self.metrics.record(scenario_name, solution)
        return solution

    def show_topology(self):
        self.topology.show_topology()

    def summary(self):
        self.metrics.summary()
