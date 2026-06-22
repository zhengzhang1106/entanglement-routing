"""
Metrics recorder for QKD experiments.

This mirrors the role of EntanglementDistribution in the original project:
record per-run metrics and export summary tables.
"""

import pandas as pd


class QKDMetrics:
    def __init__(self):
        self.records = []

    def record(self, scenario_name, solution):
        self.records.append({
            "scenario": scenario_name,
            "status": solution.get("status"),
            "objective": solution.get("objective"),
            "total_served_keys": solution.get("total_served_keys"),
            "num_activated_slots": len(solution.get("activated_links", {})),
            "final_qkp_total": sum(
                v for v in solution.get("final_qkp", {}).values()
                if v is not None
            ),
        })

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def summary(self):
        df = self.to_dataframe()
        print("\n" + "=" * 80)
        print("QKD Simulation Summary")
        print("=" * 80)
        print(df)

    def export_csv(self, output_file):
        self.to_dataframe().to_csv(output_file, index=False)
