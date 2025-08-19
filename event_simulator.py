import random
import networkx as nx
from network_request import RequestGenerator
from quantum_network import QuantumNetwork
from singlepath_routing import SPEntanglementRouting
from multipath_routing import MPGreedyEntanglementRouting
from entanglement_swapping import EntanglementSwapping
from entanglement_fusion import EntanglementFusion
from entanglement_distribution import EntanglementDistribution
from quantum_source_placement import SourcePlacement


class EventSimulator:
    def __init__(self, edge_list, memory_size, p_op, decoherence_time, num_users=3, max_timeslot=500):
        self.p_op = p_op
        self.num_users = num_users
        self.max_timeslot = max_timeslot
        self.network = QuantumNetwork(edge_list=edge_list, memory_size=memory_size, decoherence_time=decoherence_time)
        self.link_manager = self.network.entanglementlink_manager
        self.topo = self.network.topo
        self.user_gen = RequestGenerator(self.topo.get_nodes())
        self.swapping = EntanglementSwapping(self.network)
        self.fusion = EntanglementFusion(self.network)
        # self.DR = EntanglementDistribution()

    def select_center_node(self, subgraph, user_set):
        # It is possible that candidate_centers doesn't exist
        # dr_v is 0 (no entanglement link in the path)
        candidate_centers = []
        for v in self.network.nodes.values():
            if len(v.channels) >= len(user_set):
                candidate_centers.append(v.node_id)
        best_v = None
        best_dr = 0

        for v in candidate_centers:
            paths = self.get_shortest_paths_SP(v, user_set)

            dr_v = self.compute_dr_sp(paths, subgraph)
            if dr_v > best_dr:
                best_dr = dr_v
                best_v = v

        print(f"[Select Center] Chosen center node: {best_v}, with estimated DR_SP = {best_dr:.4f}")

        return best_v, best_dr

    def get_shortest_paths_SP(self, v, user_set):
        paths = {}
        # print(f"\n[Routing] Computing shortest paths from center node '{v}' to users {user_set}:")
        for s in user_set:
            if s == v:
                continue
            try:
                path = nx.shortest_path(self.topo.graph, source=v, target=s)
                paths[s] = path
                # print(f"  Path to {s}: {path}")
            except nx.NetworkXNoPath:
                paths[s] = []
                # print(f"  No path to {s}")
        return paths

    # only used for selecting center node
    def compute_dr_sp(self, paths, subgraph):
        dr = 1.0
        for path in paths.values():
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if subgraph.has_edge(u, v):
                    p_e = subgraph.edges[u, v].get("p_e", 1.0)
                    dr *= p_e
                else:
                    return 0.0
        return dr

    def run_single_trial_SP(self, user_set, p_op):
        """
        Run simulation until GHZ is formed for the given user set or until max timeslot.
        Returns number of time slots required or None if failed.
        """
        G_prime = self.link_manager.get_subgraph(current_time=1)

        # vc: The selected center node vc remains fixed
        vc, best_dr = self.select_center_node(G_prime, user_set)
        if vc is None:
            print(f"\n[Routing] Final selected center: {vc}")
            return 0, 0

        # paths: the routing solution computed for the selected central node remains fixed
        paths = self.get_shortest_paths_SP(vc, user_set)
        print(f"\n[Routing] Final selected center: {vc}")
        for s, path in paths.items():
            print(f"  {vc} -> {s}: {path}")

        SProuting = SPEntanglementRouting(self.network, user_set, p_op)
        time_to_success, cost = SProuting.sp_routing(vc, paths, self.max_timeslot)
        return time_to_success, cost

    def run_single_trial_MPG(self, user_set, p_op):
        G_prime = self.link_manager.get_subgraph(current_time=1)

        # vc: The selected center node vc remains fixed
        vc, best_dr = self.select_center_node(G_prime, user_set)
        if vc is None:
            print(f"\n[Routing] Final selected center: {vc}")
            return 0, 0

        # paths: the routing solution computed for the selected central node changes with subgraph
        MPGrouting = MPGreedyEntanglementRouting(self.network, user_set, p_op)
        time_to_success, cost = MPGrouting.mp_greedy_routing(vc, self.max_timeslot)
        return time_to_success, cost

    def run_trials(self, seed, user_sets, routing_method, dr_object):
        if seed is not None:
            random.seed(seed)

        for i, user_set in enumerate(user_sets):
            trial_num = i + 1
            print("\n" + "=" * 100)
            print(f"[Trial {trial_num} for {routing_method}] Running...")
            print(f"[user_set] {user_set}")

            self.network.reset()

            source = SourcePlacement(self.topo)
            sources = source.place_sources_for_request(user_set)
            for u, v in sources:
                self.network.attempt_entanglement(u, v, p_op=self.p_op, gen_time=0)

            self.network.show_network_status(current_time=0)

            if routing_method == 'SP':
                time_to_success, cost = self.run_single_trial_SP(user_set, self.p_op)
            elif routing_method == 'MPG':
                time_to_success, cost = self.run_single_trial_MPG(user_set, self.p_op)
            else:
                raise ValueError(f"Unknown routing_method: {routing_method}")

            if time_to_success:
                print(f"[Trial {trial_num}] ✅ GHZ generated in {time_to_success} time slots for {routing_method}.")
            else:
                print(f"[Trial {trial_num}] ❌ Failed to generate GHZ for {routing_method} within time limit.")

            dr_object.record_trial(time_to_success, cost)

            # self.DR.record_trial(time_to_success, cost)

        # self.DR.summary()


if __name__ == "__main__":
    """
        0 —— 1 —— 2
        |    |    |
        3 —— 4 —— 5
        |    |    |
        6 —— 7 —— 8
    """

    m = 3
    length = 10
    edge_list = []
    for row in range(m):
        for col in range(m):
            node = row * m + col
            # Right neighbor
            if col < m - 1:
                right = node + 1
                edge_list.append((node, right, length))
            # Bottom neighbor
            if row < m - 1:
                down = node + m
                edge_list.append((node, down, length))

    NUM_TRIALS = 500
    RANDOM_SEED = 1
    NUM_USERS = 3

    simulator = EventSimulator(edge_list, num_users=NUM_USERS, p_op=0.8, memory_size=4, decoherence_time=5, max_timeslot=10)

    dr_sp = EntanglementDistribution()
    dr_mpg = EntanglementDistribution()

    print(f"Generating {NUM_TRIALS} user sets for the experiment...")
    random.seed(RANDOM_SEED)
    user_sets_list = [simulator.user_gen.random_users(k=NUM_USERS) for _ in range(NUM_TRIALS)]
    print("\nGenerated User Sets List for all trials:")
    print(user_sets_list)

    # --- Run Simulations ---
    # Run Single-Path trials
    print("\n" + "#" * 60)
    print("###   STARTING SINGLE-PATH (SP) ROUTING SIMULATION   ###")
    print("#" * 60)
    simulator.run_trials(user_sets=user_sets_list, routing_method='SP', seed=RANDOM_SEED, dr_object=dr_sp)

    # Run Multi-Path Greedy trials
    print("\n" + "#" * 60)
    print("###   STARTING MULTI-PATH GREEDY (MPG) ROUTING SIMULATION   ###")
    print("#" * 60)
    simulator.run_trials(user_sets=user_sets_list, routing_method='MPG', seed=RANDOM_SEED, dr_object=dr_mpg)

    # --- Final Summary ---
    print("\n\n" + "*" * 50)
    print("********** FINAL SIMULATION SUMMARY   **********")
    print("*" * 50)

    print("\n--- DR Summary for SinglePath (SP) Routing ---")
    dr_sp.summary()
    print("\n--- DR Summary for MultiPath Greedy (MPG) Routing ---")
    dr_mpg.summary()