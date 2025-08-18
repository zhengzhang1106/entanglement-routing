import networkx as nx


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
        subgraph = self._approximate_steiner_tree(user_set)
        self.sources = set(subgraph.edges())
        print(f"[SourcePlacement] Sources placed on edges: {self.sources}")
        print(f"[SourcePlacement] Total cost: {self.compute_cost()}")
        return self.sources

    def _approximate_steiner_tree(self, terminals):
        """
        Approximate Steiner Tree using metric closure + MST on complete graph.
        Returns a subgraph (nx.Graph) of the original topology.
        """
        G = self.topo.graph
        # Step 1: Metric closure - complete graph with shortest path lengths
        metric_closure = nx.Graph()
        for u in terminals:
            for v in terminals:
                if u == v:
                    continue
                try:
                    length = nx.shortest_path_length(G, u, v, weight='length')
                    metric_closure.add_edge(u, v, weight=length)
                except nx.NetworkXNoPath:
                    continue

        # Step 2: MST of the complete terminal graph
        mst = nx.minimum_spanning_tree(metric_closure, weight='weight')

        # Step 3: Map MST edges back to paths in original graph
        steiner_tree = nx.Graph()
        for u, v in mst.edges():
            try:
                path = nx.shortest_path(G, source=u, target=v, weight='length')
                for i in range(len(path) - 1):
                    steiner_tree.add_edge(path[i], path[i + 1])
            except nx.NetworkXNoPath:
                continue

        return steiner_tree

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
