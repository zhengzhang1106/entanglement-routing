import networkx as nx
from steiner_tree_algorithms import approximate_steiner_tree

pair_cost = 1

class SourcePlacement:
    def __init__(self, topo):
        self.topo = topo
        self.sources = []

    def place_sources_for_request(self, user_set, method="OP", cost_budget=None, max_per_edge=1):
        """
        Args:
            user_set (list): User terminal nodes.
            method (str): 'steiner_tree' or 'all_edges'.
            cost_budget (int, optional): Total cost budget (must be even). Each source pair costs 2.
            max_per_edge (int): Maximum number of source pairs per edge.

        Returns:
            list: List of edges [(u, v), ...] where sources are deployed.
                  Each tuple represents one source pair.
        """
        # 1) Select base edges
        if method == "OP":
            subgraph = approximate_steiner_tree(self.topo.graph, user_set)
            base_edges = list(subgraph.edges())
        elif method == "NOP":
            base_edges = list(self.topo.get_edges())
        else:
            raise ValueError(f"Unknown source placement method: {method}")

        # Normalize edges as sorted tuples (undirected)
        base_keys = sorted({tuple(sorted(e[:2])) for e in base_edges})
        if not base_keys:
            self.sources = []
            print("[SourcePlacement] No candidate edges found.")
            return self.sources

        self.sources = []
        per_edge_count = {k: 0 for k in base_keys}

        # 2) If no budget: assign one pair per edge (within max_per_edge)
        if cost_budget is None:
            for (u, v) in base_keys:
                if per_edge_count[(u, v)] < max_per_edge:
                    self.sources.append((u, v))
                    per_edge_count[(u, v)] += 1
            print(f"[SourcePlacement] Method: {method}, Sources placed: {self.sources}")
            print(f"[SourcePlacement] Total cost: {self.compute_cost()}")
            print("[SourcePlacement] Cost budget: None")
            return self.sources

        # 3) With budget
        if cost_budget < 0:
            raise ValueError("cost_budget must be non-negative")
        if cost_budget % pair_cost != 0:
            print(f"[SourcePlacement][WARN] cost_budget={cost_budget} is not even, "
                  f"using {cost_budget - 1} instead.")
        budget_pairs = cost_budget // pair_cost

        capacity_pairs = len(base_keys) * max_per_edge
        target_pairs = min(budget_pairs, capacity_pairs)

        # 4) Round-robin allocation across edges
        placed = 0
        idx = 0
        n = len(base_keys)
        while placed < target_pairs:
            u, v = base_keys[idx % n]
            if per_edge_count[(u, v)] < max_per_edge:
                self.sources.append((u, v))
                per_edge_count[(u, v)] += 1
                placed += 1
            idx += 1
            if idx >= n and all(per_edge_count[k] >= max_per_edge for k in base_keys):
                break  # capacity exhausted

        # 5) Print and return
        print(f"[SourcePlacement] Method: {method}, Sources placed: {self.sources}")
        print(f"[SourcePlacement] Total cost: {self.compute_cost()} (target={2 * target_pairs})")
        print(f"[SourcePlacement] Cost budget: {cost_budget}, max_per_edge={max_per_edge}, "
              f"capacity_pairs={capacity_pairs}, used_pairs={placed}")
        return self.sources

    def compute_cost(self):
        return len(self.sources) * pair_cost


if __name__ == "__main__":
    from network_topology import Topology

    """
       0 —— 1 —— 2
       |    |    |
       3 —— 4 —— 5
       |    |    |
       6 —— 7 —— 8
       """
    edge_list = [
        (0, 1, 10),
        (0, 3, 10),
        (1, 2, 10),
        (1, 4, 10),
        (2, 5, 10),
        (3, 4, 10),
        (3, 6, 10),
        (4, 7, 10),
        (5, 8, 10),
        (6, 7, 10),
        (7, 8, 10)
    ]
    topo = Topology(edge_list)
    users = [0, 2, 7]

    print("\n" + "=" * 50 + " Steiner Tree " + "=" * 50 + "\n")
    source_placement_steiner = SourcePlacement(topo)
    sources_steiner = source_placement_steiner.place_sources_for_request(users, method="steiner_tree")

    print("\n" + "=" * 50 + " Steiner Tree with Budget " + "=" * 50 + "\n")
    source_placement_steiner_budget = SourcePlacement(topo)
    sources_steiner_budget = source_placement_steiner_budget.place_sources_for_request(users, method="steiner_tree",
                                                                                       cost_budget=10, max_per_edge=2)

    print("\n" + "=" * 50 + " All Edges with Budget " + "=" * 50 + "\n")
    source_placement_all_edges_budget = SourcePlacement(topo)
    sources_all_edges_budget = source_placement_all_edges_budget.place_sources_for_request(users, method="all_edges",
                                                                                           cost_budget=19, max_per_edge=2)