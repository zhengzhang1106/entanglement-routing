"""
Performs entanglement fusion on intermediate nodes (non-user) with multiple entangled links.
Each such node fuses its connected users into a GHZ state.
"""

from quantum_network import QuantumNetwork


class EntanglementFusion:
    def __init__(self, network):
        self.network = network
        self.link_manager = network.entanglementlink_manager

    def fuse_users(self, intermediate_node, user_list, current_time, p_op):
        # Step 1: Validate all required links exist
        for u in user_list:
            if u != intermediate_node:
                if not self._link_exists(u, intermediate_node):
                    print(f"[Fusion] Missing Bell pair: {u}–{intermediate_node}")
                    return False

        # Step 2: Remove old links and memory
        for u in user_list:
            if u != intermediate_node:
                self._remove_link(u, intermediate_node)
                self._release_memory(u, intermediate_node)
                self._release_memory(intermediate_node, u)

        # Step 3: Create new GHZ link among users
        self.link_manager.create_link(user_list, p_op=1, gen_time=current_time, length_km=0, attr="Fusion")

        # Step 4: Update memory (pairwise occupancy)
        for u in user_list:
            for v in user_list:
                if u != v:
                    self.network.nodes[u].memory.occupy_memory(v, current_time)

        print(f"[Fusion] GHZ state created among {user_list} via {intermediate_node}")
        return True

    def _link_exists(self, u, v):
        for link in self.link_manager.links:
            if u in link.nodes and v in link.nodes:
                return True
        return False

    def _remove_link(self, u, v):
        self.link_manager.links = [
            link for link in self.link_manager.links
            if not (u in link.nodes and v in link.nodes)
        ]

    def _release_memory(self, node_id, peer_id):
        mem = self.network.nodes[node_id].memory
        mem.memory_storage = {
            k: v for k, v in mem.memory_storage.items()
            if k[0] != peer_id
        }


if __name__=="__main__":
    # A —— D —— C
    #      |    |
    #      B ——
    edge_list = [("A", "D", 10), ("D", "C", 15), ("D", "B", 10), ("B", "C", 20)]
    net = QuantumNetwork(edge_list=edge_list, memory_size=4, decoherence_time=6)

    net.attempt_entanglement("A", "D", p_op=0.9, gen_time=0)
    net.attempt_entanglement("B", "D", p_op=0.9, gen_time=4)
    net.attempt_entanglement("C", "D", p_op=0.9, gen_time=4)

    print("\nBefore Fusion")
    net.show_network_status(current_time=4)
    fusion = EntanglementFusion(net)
    fusion.fuse_users("D", ["A", "B", "C"], current_time=5, p_op=0.9)
    print("\nAfter Fusion")
    net.show_network_status(current_time=5)