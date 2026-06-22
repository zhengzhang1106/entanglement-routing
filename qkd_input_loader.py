"""
CSV input loader for Satellite-HAP-GS QKD experiments.

Expected files:
- nodes.csv
- link_capacities.csv
- demands.csv
"""

import pandas as pd

from qkd_entity import QKDNode, QKDLinkCapacity, KeyDemand, NodeType, LinkType
from qkd_topology import QKDTopology


class QKDInputLoader:
    def __init__(self, nodes_file, capacities_file, demands_file):
        self.nodes_file = nodes_file
        self.capacities_file = capacities_file
        self.demands_file = demands_file

    def load_nodes(self):
        df = pd.read_csv(self.nodes_file)
        nodes = []

        for _, row in df.iterrows():
            node = QKDNode(
                node_id=str(row["node_id"]),
                node_type=NodeType(str(row["node_type"])),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                altitude_km=float(row["altitude_km"]),
                connection_limit=int(row.get("connection_limit", 1)),
            )
            nodes.append(node)

        return nodes

    def load_link_capacities(self):
        df = pd.read_csv(self.capacities_file)
        capacities = []

        for _, row in df.iterrows():
            cap = QKDLinkCapacity(
                time_slot=int(row["time_slot"]),
                u=str(row["u"]),
                v=str(row["v"]),
                link_type=LinkType(str(row["link_type"])),
                capacity_bits=float(row["capacity_bits"]),
                visible=bool(int(row.get("visible", 1))),
            )
            capacities.append(cap)

        return capacities

    def load_demands(self):
        df = pd.read_csv(self.demands_file)
        demands = []

        for _, row in df.iterrows():
            demand = KeyDemand(
                demand_id=str(row["demand_id"]),
                time_slot=int(row["time_slot"]),
                src=str(row["src"]),
                dst=str(row["dst"]),
                key_bits=float(row["key_bits"]),
            )
            demands.append(demand)

        return demands

    def build_topology(self):
        nodes = self.load_nodes()
        capacities = self.load_link_capacities()

        topo = QKDTopology()

        for node in nodes:
            topo.add_qkd_node(node)

        seen_edges = set()
        for cap in capacities:
            edge_key = cap.edge_key()
            if edge_key not in seen_edges:
                topo.add_qkd_edge(cap.u, cap.v, cap.link_type)
                seen_edges.add(edge_key)

        return topo
