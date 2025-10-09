"""
    Two methods to generate the grid topology
    Define the physical connection structure
        node
        edge
            length (float): the length of the quantum channel
"""

import networkx as nx
import matplotlib.pyplot as plt


class Topology:
    def __init__(self, edge_list):
        self.graph = nx.Graph()
        for u, v, length_km in edge_list:
            self.graph.add_node(u)
            self.graph.add_node(v)
            self.graph.add_edge(u, v, length=length_km)
    #
    # def __init__(self, x, y, length_km):
    #     self.graph = nx.grid_2d_graph(x, y)
    #     for (u, v) in self.graph.edges():
    #         self.graph.edges[u, v]['length'] = length_km

    def show_topology(self):
        print("Show the current topology:")
        print(f"  Nodes: {self.graph.nodes(data=True)}")
        print(f"  Edges: {self.graph.edges(data=True)}")
        print('\n')

    def get_edges(self):
        return list(self.graph.edges)

    def get_nodes(self):
        return list(self.graph.nodes)

    def get_edge_length(self, node1, node2):
        return self.graph.edges[node1, node2].get("length", None)

    # the attribute of the edges
    def get_edge_data(self, node1, node2):
        return self.graph.get_edge_data(node1, node2, default=0)

    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))

    def draw_topology(self):
        pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}
        labels = {node: str(node) for node in self.graph.nodes()}

        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw(self.graph, pos, ax=ax, with_labels=True, labels=labels,
                node_size=500, node_color="skyblue", font_size=8, font_color="black")

        edge_labels = {(u, v): d['length'] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=7)

        ax.set_title("Grid Topology Visualization")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    """
    0 —— 1 —— 2
    |    |    |
    3 —— 4 —— 5
    |    |    |
    6 —— 7 —— 8
    """
    # edge_list = [
    #     (0, 1, 10),
    #     (0, 3, 10),
    #     (1, 2, 10),
    #     (1, 4, 10),
    #     (2, 5, 10),
    #     (3, 4, 10),
    #     (3, 6, 10),
    #     (4, 7, 10),
    #     (5, 8, 10),
    #     (6, 7, 10),
    #     (7, 8, 10)
    # ]

    # m = 5
    # length = 10
    # edge_list = []
    # for row in range(m):
    #     for col in range(m):
    #         node = row * m + col
    #         # Right neighbor
    #         if col < m - 1:
    #             right = node + 1
    #             edge_list.append((node, right, length))
    #         # Bottom neighbor
    #         if row < m - 1:
    #             down = node + m
    #             edge_list.append((node, down, length))

    # topo = Topology(edge_list)

    # topo.show_topology()
    # print("Nodes", topo.get_nodes())
    # print("Edges", topo.get_edges())
    # print("Length of edge 0-1:", topo.get_edge_length(0, 1))
    # print("Attribute of edge 0-1", topo.get_edge_data(0, 1))
    # print("Neighbors of node 3:", topo.get_neighbors(3))

    topo = Topology(5, 5, 10)
    topo.draw_topology()
    print("The current topology:")
    topo.show_topology()
    print("Length of edge (0, 0)-(1, 0):", topo.get_edge_length((0, 0), (1, 0)))
    print("Attribute of edge (0, 0)-(1, 0):", topo.get_edge_data((0, 0), (1, 0)))
    print("Neighbors of (0, 1):", topo.get_neighbors((0, 1)))

