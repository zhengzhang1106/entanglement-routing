import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from event_simulator import EventSimulator
from entanglement_distribution import EntanglementDistribution

# --- Simulation Parameters ---
RANDOM_SEED = 1
NUM_TRIALS = 50
MAX_TIMEESLOT_PER_TRIAL = 300
DECOHERENCE_TIME = 10
MAX_PER_EDGE = 5
EDGE_LENGTH_KM = 1


def run_and_get_metrics(params):
    """
    Helper function to run a simulation and return a tuple of metrics:
    (cost_efficiency_actual, average_dr)
    """
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
        source_method=params['source_method'],
        seed=RANDOM_SEED,
        dr_object=dr_object,
        cost_budget=params['cost_budget']
    )

    return (
        dr_object.cost_efficiency_actual(),
        dr_object.average_dr()
    )


def create_dual_axis_plot(title):
    """Creates a matplotlib figure and axes with a shared x-axis and two y-axes."""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    ax1.set_xlabel('Source Budget')
    ax1.set_ylabel('Cost-Efficiency (CE)', color='tab:blue')
    ax2.set_ylabel('Distribution Rate (DR)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(title)

    return fig, ax1, ax2


def plot_protocols_vs_budget(excel_writer, output_dir):
    """Generates Plot 1: Protocols vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 1: Protocols vs. Budget   ###")
    print("#" * 60)

    protocols = ['SP', 'MPG', 'MPC', 'MPP']
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 41, 2)

    plot_data = []
    results = {f'{p} ({sm})': {'ce_actual': [], 'dr': []} for p in protocols for sm in source_methods}

    for method in source_methods:
        for protocol in protocols:
            for budget in cost_budgets:
                label = f'{protocol} ({method})'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': 5, 'width_network': 5, 'num_users': 4,
                    'p_op': 0.9, 'routing_method': protocol,
                    'source_method': method, 'cost_budget': budget
                }
                ce_actual, dr = run_and_get_metrics(params)
                results[label]['ce_actual'].append(ce_actual)
                results[label]['dr'].append(dr)

                plot_data.append({
                    'Protocol': protocol, 'Source Method': method, 'Budget': budget,
                    'Cost_Efficiency_Actual': ce_actual, 'Distribution_Rate': dr
                })

    df = pd.DataFrame(plot_data)
    df.to_excel(excel_writer, sheet_name='Protocols_vs_Budget', index=False)

    fig, ax1, ax2 = create_dual_axis_plot('Protocols vs. Source Budget: CE and DR')
    lines, labels = [], []
    for label, data in results.items():
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        l1, = ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=4)
        lines.append(l1)
        labels.append(f'{label} (CE)')

        l2, = ax2.plot(cost_budgets, data['dr'], marker='^', linestyle='--', color=color, markersize=4)
        lines.append(l2)
        labels.append(f'{label} (DR)')

    ax1.legend(lines, labels, bbox_to_anchor=(1.15, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_protocols_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 1 saved and data exported to Excel.")


def plot_mpp_robustness_vs_budget(excel_writer, output_dir):
    """Generates Plot 2: MP-P Robustness vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 2: MP-P Robustness vs. Budget   ###")
    print("#" * 60)

    p_ops = [0.7, 0.8, 0.9]
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 41, 2)

    plot_data = []
    results = {f'p_op={p} ({sm})': {'ce_actual': [], 'dr': []} for p in p_ops for sm in source_methods}

    for method in source_methods:
        for p_op in p_ops:
            for budget in cost_budgets:
                label = f'p_op={p_op} ({method})'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': 5, 'width_network': 5, 'num_users': 4,
                    'p_op': p_op, 'routing_method': 'MPP',
                    'source_method': method, 'cost_budget': budget
                }
                ce_actual, dr = run_and_get_metrics(params)
                results[label]['ce_actual'].append(ce_actual)
                results[label]['dr'].append(dr)

                plot_data.append({
                    'P_op': p_op, 'Source Method': method, 'Budget': budget,
                    'Cost_Efficiency_Actual': ce_actual, 'Distribution_Rate': dr
                })

    df = pd.DataFrame(plot_data)
    df.to_excel(excel_writer, sheet_name='MPP_Robustness', index=False)

    fig, ax1, ax2 = create_dual_axis_plot('MP-P Robustness vs. Budget: CE and DR')
    lines, labels = [], []
    for label, data in results.items():
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        l1, = ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=4)
        lines.append(l1)
        labels.append(f'{label} (CE)')

        l2, = ax2.plot(cost_budgets, data['dr'], marker='^', linestyle='--', color=color, markersize=4)
        lines.append(l2)
        labels.append(f'{label} (DR)')

    ax1.legend(lines, labels, bbox_to_anchor=(1.15, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_mpp_robustness_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 2 saved and data exported to Excel.")


def plot_mpp_scalability_network_size_vs_budget(excel_writer, output_dir):
    """Generates Plot 3: MP-P Scalability (Network Size) vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 3: MP-P Scalability (Network Size) vs. Budget   ###")
    print("#" * 60)

    network_sizes = [(3, 3), (4, 4), (5, 5)]
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 41, 2)

    plot_data = []
    results = {f'{s[0]}x{s[1]} ({sm})': {'ce_actual': [], 'dr': []} for s in network_sizes for sm in source_methods}

    for method in source_methods:
        for size in network_sizes:
            for budget in cost_budgets:
                label = f'{size[0]}x{size[1]} ({method})'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': size[0], 'width_network': size[1], 'num_users': 3,
                    'p_op': 0.9, 'routing_method': 'MPP',
                    'source_method': method, 'cost_budget': budget
                }
                ce_actual, dr = run_and_get_metrics(params)
                results[label]['ce_actual'].append(ce_actual)
                results[label]['dr'].append(dr)

                plot_data.append({
                    'Network_Size': f'{size[0]}x{size[1]}', 'Source Method': method, 'Budget': budget,
                    'Cost_Efficiency_Actual': ce_actual, 'Distribution_Rate': dr
                })

    df = pd.DataFrame(plot_data)
    df.to_excel(excel_writer, sheet_name='MPP_Scalability_Network', index=False)

    fig, ax1, ax2 = create_dual_axis_plot('MP-P Scalability (Network Size) vs. Budget: CE and DR')
    lines, labels = [], []
    for label, data in results.items():
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        l1, = ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=4)
        lines.append(l1)
        labels.append(f'{label} (CE)')

        l2, = ax2.plot(cost_budgets, data['dr'], marker='^', linestyle='--', color=color, markersize=4)
        lines.append(l2)
        labels.append(f'{label} (DR)')

    ax1.legend(lines, labels, bbox_to_anchor=(1.15, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '3_mpp_scalability_network_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 3 saved and data exported to Excel.")


def plot_mpp_scalability_users_vs_budget(excel_writer, output_dir):
    """Generates Plot 4: MP-P Scalability (Number of Users) vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 4: MP-P Scalability (Number of Users) vs. Budget   ###")
    print("#" * 60)

    num_users_list = [3, 4, 5]
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 41, 2)

    plot_data = []
    results = {f'{n} Users ({sm})': {'ce_actual': [], 'dr': []} for n in num_users_list for sm in source_methods}

    for method in source_methods:
        for num_users in num_users_list:
            for budget in cost_budgets:
                label = f'{num_users} Users ({method})'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': 5, 'width_network': 5, 'num_users': num_users,
                    'p_op': 0.9, 'routing_method': 'MPP',
                    'source_method': method, 'cost_budget': budget
                }
                ce_actual, dr = run_and_get_metrics(params)
                results[label]['ce_actual'].append(ce_actual)
                results[label]['dr'].append(dr)

                plot_data.append({
                    'Num_Users': num_users, 'Source Method': method, 'Budget': budget,
                    'Cost_Efficiency_Actual': ce_actual, 'Distribution_Rate': dr
                })

    df = pd.DataFrame(plot_data)
    df.to_excel(excel_writer, sheet_name='MPP_Scalability_Users', index=False)

    fig, ax1, ax2 = create_dual_axis_plot('MP-P Scalability (Number of Users) vs. Budget: CE and DR')
    lines, labels = [], []
    for label, data in results.items():
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        l1, = ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=4)
        lines.append(l1)
        labels.append(f'{label} (CE)')

        l2, = ax2.plot(cost_budgets, data['dr'], marker='^', linestyle='--', color=color, markersize=4)
        lines.append(l2)
        labels.append(f'{label} (DR)')

    ax1.legend(lines, labels, bbox_to_anchor=(1.15, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_mpp_scalability_users_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 4 saved and data exported to Excel.")


if __name__ == "__main__":
    # Create a unique, timestamped directory for this simulation run
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_directory = os.path.join('simulation_plots', timestamp)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Created output directory: {output_directory}")

    # Define the Excel file path with the same timestamp
    excel_filepath = f"simulation_results_{timestamp}.xlsx"

    # Create a single Excel writer object to save all dataframes to one file
    with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
        plot_protocols_vs_budget(writer, output_directory)
        plot_mpp_robustness_vs_budget(writer, output_directory)
        plot_mpp_scalability_network_size_vs_budget(writer, output_directory)
        plot_mpp_scalability_users_vs_budget(writer, output_directory)

    print("\n\n" + "*" * 50)
    print("********** ALL SIMULATIONS COMPLETE   **********")
    print(f"All plot images saved in '{output_directory}'.")
    print(f"All simulation data saved in '{excel_filepath}'.")
    print("*" * 50)