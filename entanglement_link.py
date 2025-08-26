import networkx as nx
import random
from network_topology import Topology


class EntanglementLink:
    def __init__(self, link_id, nodes, gen_time, length_km, p_op=0.9, loss_coef_dB_per_km=0.2, attr=None):
        self.link_id = link_id
        self.nodes = nodes  # list of nodes: ["A", "B"] or ["A", "B", "C"]
        self.gen_time = gen_time
        self.length_km = length_km
        self.p_op = p_op
        self.loss_coef = loss_coef_dB_per_km
        self.attr = attr

        # Calculate link loss and entanglement generation probability
        self.p_loss = self.calculate_loss()
        self.p_e = self.calculate_entanglement_prob()

    def calculate_loss(self):
        return 1 - 10 ** (-self.loss_coef * self.length_km / 10)

    def calculate_entanglement_prob(self):
        return self.p_op * (1 - self.p_loss)

    def show_entanglementlink_info(self):
        print(f"  Entanglement Link ID {self.link_id}, Nodes: {self.nodes}:")
        print(f"    Generate time: {self.gen_time}, Length_km: {self.length_km}, p_op: {self.p_op:.2f}, p_loss: {self.p_loss:.2f}, p_e: {self.p_e:.2f}, Via: {self.attr}")

    def is_active(self, current_time, decoherence_time):
        return current_time - self.gen_time < decoherence_time


class EntanglementLinkManager:
    def __init__(self, decoherence_time):
        self.decoherence_time = decoherence_time
        self.links = []
        self.subG = nx.Graph()
        self.link_id_counter = 0

    def create_link(self, nodes, gen_time, length_km, p_op, loss_coef=0.2, flag=False, attr=None):
        self.link_id_counter += 1
        link_id = self.link_id_counter
        temp_link = EntanglementLink(link_id, nodes, gen_time, length_km, p_op, loss_coef, attr)
        # "flag=True" is used only for test
        if flag:
            self.links.append(temp_link)
            success = True
        else:
            r = random.random()
            success = (r < temp_link.p_e)
            if success:
                self.links.append(temp_link)
            else:
                print(f"[Failed Entanglement link generation] between {nodes[0]} and {nodes[1]} "
                      f"at time {gen_time}. random_r={r:.4f}, p_e={temp_link.p_e:.4f}")
        return success, link_id

    def purge_expired_links(self, current_time):
        self.links = [link for link in self.links if link.is_active(current_time, self.decoherence_time)]
        # Subgraph doesn't store the GHZ state
        self.subG = nx.Graph()
        for link in self.links:
            if len(link.nodes) == 2:
                u, v = link.nodes
                self.subG.add_edge(u, v, link_id=link.link_id, gen_time=link.gen_time, p_e=round(link.p_e, 2))

    def get_subgraph(self, current_time):
        self.purge_expired_links(current_time)
        return self.subG

    def remove_links_by_nodes(self, nodes_to_remove):
        self.links = [link for link in self.links if
                      tuple(sorted(link.nodes)) not in [tuple(sorted(n)) for n in nodes_to_remove]]

    def remove_link_by_id(self, link_id):
        self.links = [link for link in self.links if link.link_id != link_id]

    def show_active_links(self, current_time):
        self.purge_expired_links(current_time)
        print(f"Active Entanglement Links at [time slot {current_time}]:")
        for links in self.links:
            links.show_entanglementlink_info()
        print('\n')
        print(f"Show the subgraph (bipartite) at [time slot {current_time}]:")
        print(f"  SubGraph Nodes: {self.subG.nodes(data=True)}")
        print(f"  SubGraph Edges: {self.subG.edges(data=True)}")
        print('\n')


if __name__ == "__main__":
    # A - B - C - D
    edge_list = [
        ("A", "B", 10),
        ("B", "C", 15),
        ("C", "D", 20)
    ]
    topo = Topology(edge_list)
    topo.show_topology()

    manager = EntanglementLinkManager(decoherence_time=6)
    manager.create_link(["A", "B"], gen_time=0, length_km=topo.get_edge_length("A", "B"), p_op=0.9)
    manager.create_link(["B", "C"], gen_time=5, length_km=topo.get_edge_length("B", "C"), p_op=0.9)
    manager.show_active_links(current_time=4)
    manager.show_active_links(current_time=8)
