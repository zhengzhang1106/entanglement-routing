import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
from event_simulator import EventSimulator
from entanglement_distribution import EntanglementDistribution
import numpy as np

# --- Simulation Parameters ---
RANDOM_SEED = 1
NUM_TRIALS = 500
MAX_TIMEESLOT_PER_TRIAL = 200
DECOHERENCE_TIME = 1
MAX_PER_EDGE = 10
EDGE_LENGTH_KM = 1


def run_and_get_metrics(params, user_sets_list):
    """
    Helper function to run a simulation and return a tuple of metrics.
    NOW ACCEPTS a list of user sets.
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

    simulator.run_trials(
        user_sets=user_sets_list,
        routing_method=params['routing_method'],
        source_method=params['source_method'],
        seed=RANDOM_SEED,
        dr_object=dr_object,
        cost_budget=params['cost_budget']
    )

    return dr_object.get_summary_dict()


def line_style_for(method: str) -> str:
    return '-' if method == 'steiner_tree' else '--'


def bar_style_kwargs(method: str, color):
    if method == 'steiner_tree':
        return dict(color=color, alpha=0.7)
    else:
        return dict(facecolor='none', edgecolor=color, hatch='//', linewidth=1.5)


def create_combo_plot(title, x_label='Source Budget'):
    """
    Creates a matplotlib figure and axes with a shared x-axis.
    Cost-Efficiency (CE) is a line plot on the left Y-axis.
    Distribution Rate (DR) is a bar plot on the right Y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Cost-Efficiency (CE)', color='tab:blue', fontsize=14)
    ax2.set_ylabel('Distribution Rate (DR)', color='tab:red', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
    ax1.set_title(title, fontsize=16)

    return fig, ax1, ax2


def plot_protocols_vs_budget(excel_writer, output_dir, user_sets_list):
    """Generates Plot 1: Protocols vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 1: Protocols vs. Budget   ###")
    print("#" * 60)

    protocols = ['SP', 'MPG', 'MPC', 'MPP']
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 31, 2)

    all_plot_data = []
    results = {f'{p}_{sm}': {'ce_actual': [], 'dr': []} for p in protocols for sm in source_methods}

    for method in source_methods:
        for protocol in protocols:
            for budget in cost_budgets:
                label = f'{protocol}_{method}'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': 5, 'width_network': 5, 'num_users': 4,
                    'p_op': 0.9, 'routing_method': protocol,
                    'source_method': method, 'cost_budget': budget
                }
                # Pass the consistent user_sets_list
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                # Prepare data for Excel export, including summary
                plot_point_data = {'Protocol': protocol, 'Source Method': method, 'Budget': budget}
                plot_point_data.update(summary_data)  # Add all summary fields
                plot_point_data['user_sets'] = str(user_sets_list)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='Protocols_vs_Budget', index=False)

    fig, ax1, ax2 = create_combo_plot('Protocols vs. Source Budget: CE and DR')

    # Calculate positions for bars
    base_x = np.array(cost_budgets, dtype=float)
    num_series = len(protocols) * len(source_methods)
    step = base_x[1] - base_x[0] if len(base_x) > 1 else 1.0
    bar_width = step / (num_series + 1)

    for i, (label, data) in enumerate(results.items()):
        color = plt.cm.get_cmap('tab10')(i)

        # Line plot for Cost-Efficiency
        ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=5,
                 label=f'{label} (CE)')

        # Bar plot for Distribution Rate
        x = base_x + (i - (num_series - 1) / 2) * bar_width
        ax2.bar(x, data['dr'], width=bar_width, color=color, alpha=0.7, label=f'{label} (DR)')

    ax2.set_xticks(base_x)
    ax2.set_xticklabels(cost_budgets)

    # Create a single legend for both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    fig.savefig(os.path.join(output_dir, '1_protocols_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 1 saved and data exported to Excel.")


def plot_mpp_robustness_vs_budget(excel_writer, output_dir, user_sets_list):
    """Generates Plot 2: MP-P Robustness vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 2: MP-P Robustness vs. Budget   ###")
    print("#" * 60)

    p_ops = [0.7, 0.8, 0.9]
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 31, 2)

    all_plot_data = []
    results = {f'p{p_op}_{sm}': {'ce_actual': [], 'dr': []} for p_op in p_ops for sm in source_methods}

    for method in source_methods:
        for p_op in p_ops:
            for budget in cost_budgets:
                label = f'p{p_op}_{method}'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': 5, 'width_network': 5, 'num_users': 4,
                    'p_op': p_op, 'routing_method': 'MPP',
                    'source_method': method, 'cost_budget': budget
                }
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {'P_op': p_op, 'Source Method': method, 'Budget': budget}
                plot_point_data.update(summary_data)
                plot_point_data['user_sets'] = str(user_sets_list)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='MPP_Operation Probability', index=False)

    fig, ax1, ax2 = create_combo_plot('MPP Operation Probability vs. Budget: CE and DR')

    # Calculate positions for bars
    base_x = np.array(cost_budgets, dtype=float)
    num_series = len(p_ops) * len(source_methods)
    step = base_x[1] - base_x[0] if len(base_x) > 1 else 1.0
    bar_width = step / (num_series + 1)

    for i, (label, data) in enumerate(results.items()):
        color = plt.cm.get_cmap('tab10')(i)
        ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=5,
                 label=f'{label} (CE)')

        x = base_x + (i - (num_series - 1) / 2) * bar_width
        ax2.bar(x, data['dr'], width=bar_width, color=color, alpha=0.7, label=f'{label} (DR)')

    ax2.set_xticks(base_x)
    ax2.set_xticklabels(cost_budgets)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(os.path.join(output_dir, '2_mpp_Operation Probability_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 2 saved and data exported to Excel.")


def plot_mpp_scalability_network_size_vs_budget(excel_writer, output_dir, base_user_sets_lists):
    """Generates Plot 3: MP-P Scalability (Network Size) vs. Budget (Refactored)"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 3: MP-P Scalability (Network Size) vs. Budget   ###")
    print("#" * 60)

    network_sizes = [(3, 3), (4, 4), (5, 5)]
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 31, 2)

    all_plot_data = []
    results = {f'{s[0]}x{s[1]}_{sm}': {'ce_actual': [], 'dr': []} for s in network_sizes for sm in source_methods}

    for method in source_methods:
        for size in network_sizes:
            user_sets_list = base_user_sets_lists[f'{size[0]}x{size[1]}']
            for budget in cost_budgets:
                label = f'{size[0]}x{size[1]}_{method}'
                print(f"\n--- Running: {size[0]}x{size[1]} ({method}), Budget={budget} ---")
                params = {
                    'length_network': size[0], 'width_network': size[1], 'num_users': 3,
                    'p_op': 0.9, 'routing_method': 'MPP',
                    'source_method': method, 'cost_budget': budget
                }
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {'Network_Size': f'{size[0]}x{size[1]}', 'Source Method': method, 'Budget': budget}
                plot_point_data.update(summary_data)
                plot_point_data['user_sets'] = str(user_sets_list)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='MPP_Scalability_Network', index=False)

    fig, ax1, ax2 = create_combo_plot('MPP Scalability (Network Size) vs. Budget: CE and DR')

    base_x = np.array(cost_budgets, dtype=float)
    num_series = len(network_sizes) * len(source_methods)
    step = base_x[1] - base_x[0] if len(base_x) > 1 else 1.0
    bar_width = step / (num_series + 1)

    for i, (label, data) in enumerate(results.items()):
        color = plt.cm.get_cmap('tab10')(i)

        ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=5,
                 label=f'{label} (CE)')

        x = base_x + (i - (num_series - 1) / 2) * bar_width
        ax2.bar(x, data['dr'], width=bar_width, color=color, alpha=0.7, label=f'{label} (DR)')

    ax2.set_xticks(base_x)
    ax2.set_xticklabels(cost_budgets)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(os.path.join(output_dir, '3_mpp_scalability_network_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 3 saved and data exported to Excel.")


def plot_mpp_scalability_users_vs_budget(excel_writer, output_dir, base_user_sets_lists):
    """Generates Plot 4: MP-P Scalability (Number of Users) vs. Budget (Refactored)"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 4: MPP Scalability (Number of Users) vs. Budget   ###")
    print("#" * 60)

    num_users_list = [3, 4, 5]
    source_methods = ["steiner_tree", "all_edges"]
    cost_budgets = range(10, 31, 2)

    all_plot_data = []
    results = {f'{n}users_{sm}': {'ce_actual': [], 'dr': []} for n in num_users_list for sm in source_methods}

    for method in source_methods:
        for num_users in num_users_list:
            # Get the correct user_sets_list for this number of users
            user_sets_list = base_user_sets_lists[f'{num_users}_users']
            for budget in cost_budgets:
                label = f'{num_users}users_{method}'
                print(f"\n--- Running: {num_users} Users ({method}), Budget={budget} ---")
                params = {
                    'length_network': 5, 'width_network': 5, 'num_users': num_users,
                    'p_op': 0.9, 'routing_method': 'MPP',
                    'source_method': method, 'cost_budget': budget
                }
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {'Num_Users': num_users, 'Source Method': method, 'Budget': budget}
                plot_point_data.update(summary_data)
                plot_point_data['user_sets'] = str(user_sets_list)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='MPP_Scalability_Users', index=False)

    fig, ax1, ax2 = create_combo_plot('MPP Scalability (Number of Users) vs. Budget: CE and DR')

    base_x = np.array(cost_budgets, dtype=float)
    num_series = len(num_users_list) * len(source_methods)
    step = base_x[1] - base_x[0] if len(base_x) > 1 else 1.0
    bar_width = step / (num_series + 1)

    for i, (label, data) in enumerate(results.items()):
        color = plt.cm.get_cmap('tab10')(i)

        ax1.plot(cost_budgets, data['ce_actual'], marker='o', linestyle='-', color=color, markersize=5,
                 label=f'{label} (CE)')

        x = base_x + (i - (num_series - 1) / 2) * bar_width
        ax2.bar(x, data['dr'], width=bar_width, color=color, alpha=0.7, label=f'{label} (DR)')

    ax2.set_xticks(base_x)
    ax2.set_xticklabels(cost_budgets)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(os.path.join(output_dir, '4_mpp_scalability_users_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 4 saved and data exported to Excel.")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_directory = os.path.join('simulation_plots', timestamp)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Created output directory: {output_directory}")

    excel_filepath = os.path.join(output_directory, f"simulation_results_{timestamp}.xlsx")

    # --- Pre-generate User Sets for Consistency ---
    # Since different plots have different parameters (num_users, network_size),
    # we need to generate user sets for each unique configuration.

    print("Pre-generating user sets for consistency...")
    random.seed(RANDOM_SEED)

    # User sets for Plot 1 and 2 (5x5 network, 4 users)
    sim_for_users_p1_p2 = EventSimulator(5, 5, EDGE_LENGTH_KM, MAX_PER_EDGE, 0.9, DECOHERENCE_TIME, num_users=3)
    user_sets_p1_p2 = [sim_for_users_p1_p2.user_gen.random_users(k=3) for _ in range(NUM_TRIALS)]

    # User sets for Plot 3 (varying network size, 3 users)
    base_user_sets_p3 = {}
    for size in [(3, 3), (4, 4), (5, 5)]:
        sim = EventSimulator(size[0], size[1], EDGE_LENGTH_KM, MAX_PER_EDGE, 0.9, DECOHERENCE_TIME, num_users=3)
        base_user_sets_p3[f'{size[0]}x{size[1]}'] = [sim.user_gen.random_users(k=3) for _ in range(NUM_TRIALS)]

    # User sets for Plot 4 (5x5 network, varying users)
    base_user_sets_p4 = {}
    sim_for_p4 = EventSimulator(5, 5, EDGE_LENGTH_KM, MAX_PER_EDGE, 0.9, DECOHERENCE_TIME, 5)  # Max users
    for num_users in [3, 4, 5]:
        base_user_sets_p4[f'{num_users}_users'] = [sim_for_p4.user_gen.random_users(k=num_users) for _ in
                                                   range(NUM_TRIALS)]

    print("User set generation complete.")

    with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
        plot_protocols_vs_budget(writer, output_directory, user_sets_p1_p2)
        plot_mpp_robustness_vs_budget(writer, output_directory, user_sets_p1_p2)
        plot_mpp_scalability_network_size_vs_budget(writer, output_directory, base_user_sets_p3)
        plot_mpp_scalability_users_vs_budget(writer, output_directory, base_user_sets_p4)

    print("\n\n" + "*" * 50)
    print("********** ALL SIMULATIONS COMPLETE   **********")
    print(f"All plot images and data saved in '{output_directory}'.")
    print("*" * 50)