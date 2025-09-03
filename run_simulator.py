import random
import matplotlib.pyplot as plt
import os
from event_simulator import EventSimulator
from entanglement_distribution import EntanglementDistribution

# --- Simulation Parameters ---
RANDOM_SEED = 1
NUM_TRIALS = 50
MAX_TIMEESLOT_PER_TRIAL = 300
DECOHERENCE_TIME = 10
MAX_PER_EDGE = 5
EDGE_LENGTH_KM = 1
SOURCE_METHOD = "steiner_tree"


def run_and_get_cost_efficiency(params):
    """Helper function to run a simulation and return cost-efficiency."""
    simulator = EventSimulator(
        length_network=params['length_network'],
        width_network=params['width_network'],
        edge_length_km=EDGE_LENGTH_KM,
        num_users=params['num_users'],
        p_op=params['p_op'],
        max_per_edge=MAX_PER_EDGE,
        decoherence_time=DECOHERENCE_TIME,
        max_timeslot=MAX_TIMEESLOT_PER_TRIAL
    )

    dr_object = EntanglementDistribution()

    random.seed(RANDOM_SEED)
    user_sets_list = [simulator.user_gen.random_users(k=params['num_users']) for _ in range(NUM_TRIALS)]

    simulator.run_trials(
        user_sets=user_sets_list,
        routing_method=params['routing_method'],
        source_method=SOURCE_METHOD,
        seed=RANDOM_SEED,
        dr_object=dr_object,
        cost_budget=params['cost_budget']
    )

    return dr_object.cost_effiency()


def plot_protocols_vs_budget():
    """
    Generates Plot 1: Cost-efficiency of entanglement distribution protocols vs. source budget.
    """
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 1: Protocols vs. Budget   ###")
    print("#" * 60)

    protocols = ['SP', 'MPG', 'MPC', 'MPP']
    cost_budgets = range(10, 41, 5)
    results = {proto: [] for proto in protocols}

    for protocol in protocols:
        for budget in cost_budgets:
            print(f"\n--- Running: Protocol={protocol}, Budget={budget} ---")
            params = {
                'length_network': 5,
                'width_network': 5,
                'num_users': 4,
                'p_op': 0.9,
                'routing_method': protocol,
                'cost_budget': budget
            }
            efficiency = run_and_get_cost_efficiency(params)
            results[protocol].append(efficiency)

    plt.figure(figsize=(10, 6))
    for protocol, efficiencies in results.items():
        plt.plot(cost_budgets, efficiencies, marker='o', linestyle='-', label=protocol)

    plt.title('Cost-Efficiency of Protocols vs. Source Budget')
    plt.xlabel('Source Budget')
    plt.ylabel('Cost-Efficiency (GHZ states / timeslot / cost)')
    plt.grid(True)
    plt.legend()
    if not os.path.exists('simulation_plots'):
        os.makedirs('simulation_plots')
    plt.savefig('simulation_plots/1_protocols_vs_budget.png')
    plt.close()
    print("\n[SUCCESS] Plot 1 saved to simulation_plots/1_protocols_vs_budget.png")


def plot_mpp_robustness_vs_budget():
    """
    Generates Plot 2: Cost-efficiency of MP-P protocol with varying p_op vs. source budget.
    """
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 2: MP-P Robustness vs. Budget   ###")
    print("#" * 60)

    p_ops = [0.7, 0.8, 0.9]
    cost_budgets = range(10, 41, 5)
    results = {p_op: [] for p_op in p_ops}

    for p_op in p_ops:
        for budget in cost_budgets:
            print(f"\n--- Running: p_op={p_op}, Budget={budget} ---")
            params = {
                'length_network': 5,
                'width_network': 5,
                'num_users': 4,
                'p_op': p_op,
                'routing_method': 'MPP',
                'cost_budget': budget
            }
            efficiency = run_and_get_cost_efficiency(params)
            results[p_op].append(efficiency)

    plt.figure(figsize=(10, 6))
    for p_op, efficiencies in results.items():
        plt.plot(cost_budgets, efficiencies, marker='o', linestyle='-', label=f'p_op = {p_op}')

    plt.title('Robustness of MP-P Protocol vs. Source Budget')
    plt.xlabel('Source Budget')
    plt.ylabel('Cost-Efficiency')
    plt.grid(True)
    plt.legend()
    if not os.path.exists('simulation_plots'):
        os.makedirs('simulation_plots')
    plt.savefig('simulation_plots/2_mpp_robustness_vs_budget.png')
    plt.close()
    print("\n[SUCCESS] Plot 2 saved to simulation_plots/2_mpp_robustness_vs_budget.png")


def plot_mpp_scalability_network_size_vs_budget():
    """
    Generates Plot 3: Cost-efficiency of MP-P protocol in grid networks of different sizes vs. source budget.
    """
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 3: MP-P Scalability (Network Size) vs. Budget   ###")
    print("#" * 60)

    network_sizes = [(3, 3), (4, 4), (5, 5)] # Using (length, width) tuples
    cost_budgets = range(10, 41, 5)
    results = {f'{size[0]}x{size[1]}': [] for size in network_sizes}

    for size in network_sizes:
        size_label = f'{size[0]}x{size[1]}'
        for budget in cost_budgets:
            print(f"\n--- Running: Network Size={size_label}, Budget={budget} ---")
            params = {
                'length_network': size[0],
                'width_network': size[1],
                'num_users': 3,
                'p_op': 0.9,
                'routing_method': 'MPP',
                'cost_budget': budget
            }
            efficiency = run_and_get_cost_efficiency(params)
            results[size_label].append(efficiency)

    plt.figure(figsize=(10, 6))
    for size_label, efficiencies in results.items():
        plt.plot(cost_budgets, efficiencies, marker='o', linestyle='-', label=f'Network {size_label}')

    plt.title('Scalability of MP-P (Network Size) vs. Source Budget')
    plt.xlabel('Source Budget')
    plt.ylabel('Cost-Efficiency')
    plt.grid(True)
    plt.legend()
    if not os.path.exists('simulation_plots'):
        os.makedirs('simulation_plots')
    plt.savefig('simulation_plots/3_mpp_scalability_network_vs_budget.png')
    plt.close()
    print("\n[SUCCESS] Plot 3 saved to simulation_plots/3_mpp_scalability_network_vs_budget.png")


def plot_mpp_scalability_users_vs_budget():
    """
    Generates Plot 4: Cost-efficiency of MP-P protocol with varying numbers of users vs. source budget.
    """
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 4: MP-P Scalability (Number of Users) vs. Budget   ###")
    print("#" * 60)

    num_users_list = [3, 4, 5]
    cost_budgets = range(10, 41, 5)
    results = {f'{n} Users': [] for n in num_users_list}

    for num_users in num_users_list:
        users_label = f'{num_users} Users'
        for budget in cost_budgets:
            print("\n--- Running: Num Users={num_users}, Budget={budget} ---")
            params = {
                'length_network': 5,
                'width_network': 5,
                'num_users': num_users,
                'p_op': 0.9,
                'routing_method': 'MPP',
                'cost_budget': budget
            }
            efficiency = run_and_get_cost_efficiency(params)
            results[users_label].append(efficiency)

    plt.figure(figsize=(10, 6))
    for users_label, efficiencies in results.items():
        plt.plot(cost_budgets, efficiencies, marker='o', linestyle='-', label=users_label)

    plt.title('Scalability of MP-P (Number of Users) vs. Source Budget')
    plt.xlabel('Source Budget')
    plt.ylabel('Cost-Efficiency')
    plt.grid(True)
    plt.legend()
    if not os.path.exists('simulation_plots'):
        os.makedirs('simulation_plots')
    plt.savefig('simulation_plots/4_mpp_scalability_users_vs_budget.png')
    plt.close()
    print("\n[SUCCESS] Plot 4 saved to simulation_plots/4_mpp_scalability_users_vs_budget.png")


if __name__ == "__main__":
    # --- Run all simulations and generate plots ---
    plot_protocols_vs_budget()
    plot_mpp_robustness_vs_budget()
    plot_mpp_scalability_network_size_vs_budget()
    plot_mpp_scalability_users_vs_budget()

    print("\n\n" + "*" * 50)
    print("********** ALL SIMULATIONS COMPLETE   **********")
    print("*" * 50)