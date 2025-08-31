import random
from event_simulator import EventSimulator
from entanglement_distribution import EntanglementDistribution
from quantum_source_placement import SourcePlacement


def run_simulation_for_graph(routing_method, source_method, cost_budget, simulator_params):
    """
    Runs the simulation for a specific routing and source placement combination.
    """
    dr_object = EntanglementDistribution()

    # The simulator parameters need to be consistent across all runs
    simulator = EventSimulator(edge_list=simulator_params['EDGE_LIST'], max_per_edge=simulator_params['MAX_PER_EDGE'],
                               p_op=simulator_params['P_OP'], decoherence_time=simulator_params['DECOHERENCE_TIME'],
                               num_users=simulator_params['NUM_USERS'], max_timeslot=simulator_params['MAX_TIMESLOT'])

    # Generate user sets for all trials
    random.seed(simulator_params['RANDOM_SEED'])
    user_sets_list = [simulator.user_gen.random_users(k=simulator_params['NUM_USERS']) for _ in
                      range(simulator_params['NUM_TRIALS'])]

    print(f"\n--- Running {routing_method} with {source_method} source placement and cost_budget={cost_budget} ---")

    # Loop through each trial
    for user_set in user_sets_list:
        simulator.network.reset()

        source_placement = SourcePlacement(simulator.topo)
        sources = source_placement.place_sources_for_request(user_set, method=source_method, cost_budget=cost_budget,
                                                 max_per_edge=simulator_params['MAX_PER_EDGE'])
        cost = source_placement.compute_cost()

        num_ghz = 1
        time_to_success = 0

        # A temporary bug fix to ensure sources are placed correctly before routing starts
        # This should be handled in the routing protocols themselves in future versions
        for u, v in sources:
            simulator.network.attempt_entanglement(u, v, p_op=simulator_params['P_OP'], gen_time=0)

        if routing_method == 'SP':
            time_to_success, _ = simulator.run_single_trial_SP(user_set, simulator_params['P_OP'])
        elif routing_method == 'MPG':
            time_to_success, _ = simulator.run_single_trial_MPG(user_set, simulator_params['P_OP'])
        elif routing_method == 'MPC':
            time_to_success, _ = simulator.run_single_trial_MPC(user_set, simulator_params['P_OP'])
        elif routing_method == 'MPP':
            time_to_success, _, num_ghz = simulator.run_single_trial_MPP(user_set, simulator_params['P_OP'])

        dr_object.record_trial(time_to_success, cost, num_ghz=num_ghz)

    # Calculate average cost and cost efficiency for this cost budget
    avg_cost = dr_object.average_cost()
    avg_dr = dr_object.average_dr()
    cost_efficiency = avg_dr / avg_cost if avg_cost > 0 else 0

    return avg_cost, avg_dr, cost_efficiency


def generate_data_for_figure_1():
    """
    Main function to orchestrate the data generation for Figure 1.
    """
    # Simulation Parameters
    simulator_params = {
        'M': 4,
        'P_OP': 0.8,
        'DECOHERENCE_TIME': 3,
        'NUM_USERS': 3,
        'NUM_TRIALS': 200,
        'MAX_PER_EDGE': 2,
        'RANDOM_SEED': 1,
        'MAX_TIMESLOT': 200
    }

    m = simulator_params['M']
    edge_list = []
    for row in range(m):
        for col in range(m):
            node = row * m + col
            if col < m - 1:
                edge_list.append((node, node + 1, 10))
            if row < m - 1:
                edge_list.append((node, node + m, 10))
    simulator_params['EDGE_LIST'] = edge_list

    # Define the range of cost budgets to sweep
    # The range is based on the problem statement and typical network sizes
    cost_budgets = list(range(10, 40, 4))

    # Data storage
    data = {
        'SP_Steiner': {'costs': [], 'efficiencies': []},
        'SP_AllEdges': {'costs': [], 'efficiencies': []},
        'MPP_Steiner': {'costs': [], 'efficiencies': []},
        'MPP_AllEdges': {'costs': [], 'efficiencies': []},
    }

    # Run simulations for each protocol and source method combination
    for cost_budget in cost_budgets:
        # SP with Steiner Tree
        avg_cost, _, cost_efficiency = run_simulation_for_graph('SP', 'steiner_tree', cost_budget, simulator_params)
        data['SP_Steiner']['costs'].append(avg_cost)
        data['SP_Steiner']['efficiencies'].append(cost_efficiency)

        # SP with All Edges
        avg_cost, _, cost_efficiency = run_simulation_for_graph('SP', 'all_edges', cost_budget, simulator_params)
        data['SP_AllEdges']['costs'].append(avg_cost)
        data['SP_AllEdges']['efficiencies'].append(cost_efficiency)

        # MP-Packed with Steiner Tree
        avg_cost, _, cost_efficiency = run_simulation_for_graph('MPP', 'steiner_tree', cost_budget, simulator_params)
        data['MPP_Steiner']['costs'].append(avg_cost)
        data['MPP_Steiner']['efficiencies'].append(cost_efficiency)

        # MP-Packed with All Edges
        avg_cost, _, cost_efficiency = run_simulation_for_graph('MPP', 'all_edges', cost_budget, simulator_params)
        data['MPP_AllEdges']['costs'].append(avg_cost)
        data['MPP_AllEdges']['efficiencies'].append(cost_efficiency)

    # Print the final data
    print("\n\n" + "*" * 20 + " SIMULATION DATA FOR PLOTTING " + "*" * 20)
    for key, val in data.items():
        print(f"\nData for {key}:")
        print(f"  Costs: {val['costs']}")
        print(f"  Efficiencies: {val['efficiencies']}")


if __name__ == "__main__":
    generate_data_for_figure_1()