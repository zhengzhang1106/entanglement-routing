import networkx as nx
from steiner_tree_algorithms import approximate_steiner_tree


class SourcePlacement:
    def __init__(self, topo):
        self.topo = topo
        self.sources = []

    def place_sources_for_request(self, user_set, method="steiner_tree", cost_budget=None, max_per_edge=1):
        """
        Input: user_set - list of nodes (user terminals)
        Output: sources - list of edges [(u, v), ...] to deploy quantum sources
        Strategy:
            - method='steiner_tree': Construct Steiner Tree (approx) connecting users.
            - method='all_edges': Place source on all edges in the network.
            - total_cost (optional): Total budget for sources.
            - max_per_edge (optional): Maximum number of source pairs per edge.
        """
        if method == "steiner_tree":
            subgraph = approximate_steiner_tree(self.topo.graph, user_set)
            base_edges = list(set(subgraph.edges()))
        elif method == "all_edges":
            base_edges = list(set(self.topo.get_edges()))
        else:
            raise ValueError(f"Unknown source placement method: {method}")

        self.sources = []
        source_count = {}

        if cost_budget is None:
            # If no total_cost is provided, use the simple method (one source per edge)
            self.sources = base_edges
        else:
            # Place one source on each base edge first, within the budget
            for u, v in base_edges:
                if len(self.sources) * 2 + 2 > cost_budget:
                    break
                self.sources.append((u, v))
                edge_key = tuple(sorted((u, v)))
                source_count[edge_key] = 1

            # Add more sources to existing edges until budget or max_per_edge is reached
            while len(self.sources) * 2 < cost_budget:
                added_source = False
                for u, v in base_edges:
                    if len(self.sources) * 2 + 2 > cost_budget:
                        break
                    edge_key = tuple(sorted((u, v)))
                    if source_count.get(edge_key, 0) < max_per_edge:
                        self.sources.append((u, v))
                        source_count[edge_key] = source_count.get(edge_key, 0) + 1
                        added_source = True
                if not added_source:
                    break

        print(f"[SourcePlacement] Method: {method}, Sources placed on edges: {self.sources}")
        print(f"[SourcePlacement] Total cost: {self.compute_cost()}")
        print(f"[SourcePlacement] Cost budget:{cost_budget}")
        return self.sources

    def compute_cost(self):
        return len(self.sources) * 2


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