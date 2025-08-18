"""
Node A --- Node B --- Node C
If A-B and B-C have entanglement links, then B performs Bell measurement (swapping)
Each swap replaces two links with one, and updates node memory & link manager.
"""

from quantum_network import QuantumNetwork


class EntanglementSwapping:
    def __init__(self, network):
        self.network = network
        self.link_manager = network.entanglementlink_manager

    def entanglement_swapping(self, path, current_time, p_op):
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self._link_exists(u, v):
                print(f"[Swapping] Missing entanglement link between {u} and {v}")
                return False

        path_copy = path[:]

        while len(path_copy) >= 3:
            a = path_copy[0]
            b = path_copy[1]
            c = path_copy[2]

            nodeA = self.network.nodes[a]
            nodeB = self.network.nodes[b]
            nodeC = self.network.nodes[c]
            link_manager = self.network.entanglementlink_manager

            # 1. Remove old links A-B and B-C from link list
            link_manager.links = [
                link for link in link_manager.links
                if not ((a in link.nodes and b in link.nodes) or (b in link.nodes and c in link.nodes))
            ]

            # 2. Release memory
            # node A (with peer B)
            # intermediate node B
            # node C (with peer B)
            memoryA = nodeA.memory
            memoryA.memory_storage = {
                k: v for k, v in memoryA.memory_storage.items()
                if k[0] != b
            }

            memoryB = nodeB.memory
            memoryB.memory_storage = {
                k: v for k, v in memoryB.memory_storage.items()
                if not (k[0] == a or k[0] == c)
            }

            memoryC = nodeC.memory
            memoryC.memory_storage = {
                k: v for k, v in memoryC.memory_storage.items()
                if k[0] != b
            }

            # 3. Create new entanglement link A-C. Occupy memory of nodeA and nodeC
            self.network.attempt_entanglement(a, c, p_op=p_op, gen_time=current_time, attr="Swapping")

            print(f"[Swapping] Performed swapping at node {b}: {a} <-> {c} (via {b})")

            path_copy.pop(1)  # remove B

        return True

    def _link_exists(self, u, v):
        for link in self.link_manager.links:
            if u in link.nodes and v in link.nodes:
                return True
        return False


if __name__ == "__main__":
    # A - B - C
    edge_list = [("A", "B", 10), ("B", "C", 15), ("C", "D", 10)]
    net = QuantumNetwork(edge_list=edge_list, memory_size=4, decoherence_time=6)

    net.attempt_entanglement("A", "B", gen_time=0, p_op=0.9)
    net.attempt_entanglement("B", "C", gen_time=4, p_op=0.9)
    net.attempt_entanglement("C", "D", gen_time=4, p_op=0.9)

    print("\nBefore Swapping")
    net.show_network_status(current_time=4)
    swapping = EntanglementSwapping(net)
    success = swapping.entanglement_swapping(path=["A", "B", "C", "D"], current_time=5, p_op=0.9)
    if success:
        print(f"\nAfter Successful Swapping")
    else:
        print(f"\nAfter Failed Swapping")
    net.show_network_status(current_time=5)