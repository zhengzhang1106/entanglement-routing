"""
Topology abstraction for Satellite-HAP-GS QKD.

The style follows network_topology.py in this repository:
- networkx.Graph is used internally.
- simple accessor methods are provided for other modules.
"""

import networkx as nx
import matplotlib.pyplot as plt

from qkd_entity import QKDNode, LinkType


class QKDTopology:
    def __init__(self):
        self.graph = nx.Graph()

    def add_qkd_node(self, node: QKDNode):
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type.value,
            lat=node.lat,
            lon=node.lon,
            altitude_km=node.altitude_km,
            connection_limit=node.connection_limit,
        )

    def add_qkd_edge(self, u: str, v: str, link_type: LinkType):
        self.graph.add_edge(u, v, link_type=link_type.value)

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_edges(self):
        return list(self.graph.edges)

    def get_node_type(self, node_id: str):
        return self.graph.nodes[node_id].get("node_type")

    def get_connection_limit(self, node_id: str):
        return self.graph.nodes[node_id].get("connection_limit", 1)

    def get_neighbors(self, node_id: str):
        return list(self.graph.neighbors(node_id))

    def get_edge_key(self, u: str, v: str):
        return tuple(sorted((u, v)))

    def get_gs_nodes(self):
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "GS"]

    def get_sat_nodes(self):
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "SAT"]

    def get_hap_nodes(self):
        return [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "HAP"]

    def show_topology(self):
        print("Show QKD topology:")
        print(f"  Nodes: {self.graph.nodes(data=True)}")
        print(f"  Edges: {self.graph.edges(data=True)}")
        print("\n")

    def draw_topology(self):
        pos = nx.spring_layout(self.graph, seed=1)
        labels = {node: str(node) for node in self.graph.nodes()}

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(
            self.graph,
            pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_size=600,
            font_size=8,
        )

        edge_labels = {
            (u, v): d.get("link_type", "")
            for u, v, d in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=7)

        ax.set_title("Satellite-HAP-GS QKD Topology")
        plt.tight_layout()
        plt.show()
