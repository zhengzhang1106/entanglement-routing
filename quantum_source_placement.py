import networkx as nx
from steiner_tree_algorithms import approximate_steiner_tree


class SourcePlacement:
    def __init__(self, topo):
        self.topo = topo
        self.sources = set()

    def place_sources_for_request(self, user_set):
        """
        Input: user_set - list of nodes (user terminals)
        Output: sources - set of edges [(u, v), ...] to deploy quantum sources
        Strategy:
            - Construct Steiner Tree (approx) connecting users
            - Place source on each edge in the tree
        """
        subgraph = approximate_steiner_tree(self.topo.graph, user_set)
        self.sources = set(subgraph.edges())
        print(f"[SourcePlacement] Sources placed on edges: {self.sources}")
        print(f"[SourcePlacement] Total cost: {self.compute_cost()}")
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

    source = SourcePlacement(topo)
    sources = source.place_sources_for_request(users)
