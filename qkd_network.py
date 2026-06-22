"""
QKD network state manager.

This class wraps topology, time-indexed link capacities, and QKP states.
It plays a similar role to QuantumNetwork in the entanglement-routing codebase.
"""

from collections import defaultdict


class QKDNetwork:
    def __init__(self, topology, link_capacities, initial_qkp=None):
        self.topology = topology
        self.link_capacities = {}
        self.time_slots = set()

        for cap in link_capacities:
            key = (cap.time_slot, cap.edge_key())
            self.link_capacities[key] = cap
            self.time_slots.add(cap.time_slot)

        self.time_slots = sorted(self.time_slots)

        self.qkp = defaultdict(float)
        if initial_qkp:
            for edge_key, value in initial_qkp.items():
                self.qkp[tuple(sorted(edge_key))] = float(value)

    def get_capacity(self, time_slot, u, v):
        edge_key = tuple(sorted((u, v)))
        cap = self.link_capacities.get((time_slot, edge_key))
        if cap is None:
            return 0.0
        if not cap.visible:
            return 0.0
        return cap.capacity_bits

    def get_visible_edges(self, time_slot):
        edges = []
        for (t, edge_key), cap in self.link_capacities.items():
            if t == time_slot and cap.visible and cap.capacity_bits > 0:
                edges.append(edge_key)
        return edges

    def get_all_edges(self):
        return list(self.topology.get_edges())

    def get_all_nodes(self):
        return list(self.topology.get_nodes())

    def get_qkp(self, u, v):
        return self.qkp[tuple(sorted((u, v)))]

    def update_qkp(self, u, v, delta_bits):
        edge_key = tuple(sorted((u, v)))
        self.qkp[edge_key] += delta_bits
        if self.qkp[edge_key] < -1e-9:
            raise ValueError(f"QKP on edge {edge_key} becomes negative.")

    def reset_qkp(self):
        self.qkp = defaultdict(float)

    def show_network_status(self, time_slot):
        print("\n" + "-" * 80)
        print(f"QKD Network Status at [Time Slot {time_slot}]")
        print(f"Visible edges: {self.get_visible_edges(time_slot)}")
        print(f"QKP: {dict(self.qkp)}")
        print("-" * 80)
