"""
Baseline heuristics for Satellite-HAP-GS QKD.

First baseline:
- At each time slot, greedily activate the highest-capacity feasible links
  while respecting node connection limits.
"""

from collections import defaultdict


class GreedyQKDLinkSelection:
    def __init__(self, qkd_network):
        self.network = qkd_network
        self.topology = qkd_network.topology

    def select_links(self):
        selected = defaultdict(list)

        for t in self.network.time_slots:
            candidate_edges = []
            for u, v in self.network.get_all_edges():
                cap = self.network.get_capacity(t, u, v)
                if cap > 0:
                    candidate_edges.append((cap, tuple(sorted((u, v)))))

            candidate_edges.sort(reverse=True, key=lambda x: x[0])
            node_connection_count = defaultdict(int)

            for cap, e in candidate_edges:
                u, v = e
                u_limit = self.topology.get_connection_limit(u)
                v_limit = self.topology.get_connection_limit(v)

                if node_connection_count[u] >= u_limit:
                    continue
                if node_connection_count[v] >= v_limit:
                    continue

                selected[t].append(e)
                node_connection_count[u] += 1
                node_connection_count[v] += 1

        return dict(selected)
