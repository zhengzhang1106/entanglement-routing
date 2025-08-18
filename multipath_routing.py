"""
Implements SinglePath Routing algorithm (SP) from the paper:
"Multiuser Entanglement Distribution in Quantum Networks Using Multipath Routing"

Core steps:
 1. Select center node (vc) among user set S
 2. Compute shortest paths from vc to all s in S'
 3. While GHZ not established:
     - Simulate entanglement links (via EntanglementLinkManager)
     - Build entanglement subgraph G'
     - For each user s not yet connected to vc:
         - If path exists: perform entanglement swapping along path
     - If all s connected to vc: perform entanglement fusion

"""
import networkx as nx
from quantum_network import QuantumNetwork
from entanglement_swapping import EntanglementSwapping
from entanglement_fusion import EntanglementFusion
from quantum_source_placement import SourcePlacement


class MPGreedyEntanglementRouting:
    def __init__(self, network, user_set, p_op):
        self.p_op = p_op
        self.network = network
        self.G = self.network.topo  # original physical topology
        self.link_manager = network.entanglementlink_manager
        self.user_set = user_set
        self.swapping = EntanglementSwapping(self.network)
        self.fusion = EntanglementFusion(self.network)

    def simulate_entanglement_links(self, deployed_sources, time_slot):
        for u, v in deployed_sources:
            if self._has_entanglement_link(u, v):
                continue
            self.network.attempt_entanglement(u, v, p_op=self.p_op, gen_time=time_slot)

    def _has_entanglement_link(self, u, v):
        for link in self.link_manager.links:
            if u in link.nodes and v in link.nodes:
                return True
        return False

    def has_shared_bell_pair(self, user, vc):
        mem = self.network.nodes[user].memory.memory_storage
        return any(peer == vc for (peer, _) in mem)

    def get_shortest_paths_MP(self, v, subgraph):
        paths = {}
        # print(f"\n[Routing] Computing shortest paths from center node '{v}' to users {user_set}:")
        for s in self.user_set:
            if s == v:
                continue
            try:
                path = nx.shortest_path(subgraph, source=v, target=s)
                paths[s] = path
                # print(f"  Path to {s}: {path}")
            except nx.NetworkXNoPath:
                paths[s] = []
                # print(f"  No path to {s}")
        return paths

    def mp_greedy_routing(self, vc, max_timeslot):
        time_slot = 0
        hasGHZ = False
        source = SourcePlacement(self.network.topo)
        deployed_sources = source.place_sources_for_request(self.user_set)
        cost = source.compute_cost()

        while not hasGHZ:
            time_slot = time_slot + 1
            print("\n")
            print(f"[MultiplePathGreedy] [Time slot {time_slot}]")
            if time_slot >= max_timeslot:
                time_slot = 0
                break

            # Step 1: Attempt to generate entanglement links over all edges in R
            self.network.purge_all_expired(time_slot)
            self.simulate_entanglement_links(deployed_sources, time_slot)
            # self.network.show_network_status(current_time=time_slot)

            G_prime = self.link_manager.get_subgraph(current_time=time_slot)

            paths = self.get_shortest_paths_MP(vc, G_prime)
            print(f"\n[Routing] Selected center: {vc}")
            for s, path in paths.items():
                print(f"  {vc} -> {s}: {path}")

            # Step 2: For users who do not yet share a Bell pair with center, do swapping
            S_prime = [u for u in self.user_set if u != vc and not self.has_shared_bell_pair(u, vc)]
            for s in S_prime:
                path = paths.get(s, [])
                if path:
                    self.swapping.entanglement_swapping(path=path, current_time=time_slot, p_op=self.p_op)

            # Step 3: If all users now share Bell pairs with center node, do fusion
            remote_users = [u for u in self.user_set if u != vc]
            if all(self.has_shared_bell_pair(u, vc) for u in remote_users):
                success = self.fusion.fuse_users(vc, user_list=self.user_set, current_time=time_slot, p_op=self.p_op)
                if success:
                    print(f"[Fusion] GHZ generated at vc={vc}")
                    hasGHZ = True

        return time_slot, cost


if __name__ == "__main__":
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
        (4, 5, 10),
        (4, 7, 10),
        (5, 8, 10),
        (6, 7, 10),
        (7, 8, 10)
    ]
    users = [0, 2, 7]
    vc = 1
    max_timeslot = 50
    paths = {0: [1, 0],
             2: [1, 2],
             7: [1, 4, 7]}

    net = QuantumNetwork(edge_list=edge_list, memory_size=4, decoherence_time=6)

    source = SourcePlacement(net.topo)
    sources = source.place_sources_for_request(users)
    for u, v in sources:
        net.attempt_entanglement(u, v, p_op=0.9, gen_time=0, flag=True)

    net.show_network_status(current_time=0)

    MPGrouting = MPGreedyEntanglementRouting(net, users, p_op=0.9)
    print("\n[MultiPath_G+ Routing Test]")
    final_time, cost = MPGrouting.mp_greedy_routing(vc, max_timeslot)
    if final_time:
        print(f"[SUCCESS] GHZ state generated at time slot {final_time}")
    else:
        print("[FAILURE] Protocol did not succeed within time limit")

    net.show_network_status(current_time=final_time)
