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
NUM_TRIALS = 100
MAX_TIMEESLOT_PER_TRIAL = 100

DECOHERENCE_TIME = 1
MAX_PER_EDGE = 8
EDGE_LENGTH_KM = 10

COST_BUDGETS = list(range(20, 50, 3))

LENGTH_NETWORK_PROTOCOLS_1 = 5
WIDTH_NETWORK_PROTOCOLS_1 = 5
NUM_USERS_PROTOCOLS_1 = 3
OP_PROTOCOLS_1 = 0.8

LENGTH_NETWORK_OP_2 = 5
WIDTH_NETWORK_OP_2 = 5
NUM_USERS_OP_2 = 3

NUM_USERS_NET_SIZE_3 = 3
OP_NET_SIZE_3 = 0.8

LENGTH_NETWORK_USER_4 = 3
WIDTH_NETWORK_USER_4 = 3
OP_USER_4 = 0.8

LENGTH_NETWORK_DT_5 = 5
WIDTH_NETWORK_DT_5 = 5
NUM_USERS_DT_5 = 3
OP_DT_5 = 0.8

def export_run_parameters(writer):
    """Write key run parameters into a dedicated Excel sheet."""
    rows = [
        ("RANDOM_SEED", RANDOM_SEED),
        ("NUM_TRIALS", NUM_TRIALS),
        ("MAX_TIMEESLOT_PER_TRIAL", MAX_TIMEESLOT_PER_TRIAL),
        ("DECOHERENCE_TIME",DECOHERENCE_TIME),
        ("MAX_PER_EDGE",MAX_PER_EDGE),
        ("EDGE_LENGTH_KM",EDGE_LENGTH_KM),
        ("COST_BUDGETS", COST_BUDGETS),

        # Fig.1 基础参数
        ("LENGTH_NETWORK_PROTOCOLS_1", LENGTH_NETWORK_PROTOCOLS_1, "Fig.1"),
        ("WIDTH_NETWORK_PROTOCOLS_1",  WIDTH_NETWORK_PROTOCOLS_1,  "Fig.1"),
        ("NUM_USERS_PROTOCOLS_1",      NUM_USERS_PROTOCOLS_1,      "Fig.1"),
        ("OP_PROTOCOLS_1",             OP_PROTOCOLS_1,             "Fig.1"),
        # Fig.2
        ("LENGTH_NETWORK_OP_2",        LENGTH_NETWORK_OP_2,        "Fig.2"),
        ("WIDTH_NETWORK_OP_2",         WIDTH_NETWORK_OP_2,         "Fig.2"),
        ("NUM_USERS_OP_2",             NUM_USERS_OP_2,             "Fig.2"),
        # Fig.3
        ("NUM_USERS_NET_SIZE_3",       NUM_USERS_NET_SIZE_3,       "Fig.3"),
        ("OP_NET_SIZE_3",              OP_NET_SIZE_3,              "Fig.3"),
        # Fig.4
        ("LENGTH_NETWORK_USER_4",      LENGTH_NETWORK_USER_4,      "Fig.4"),
        ("WIDTH_NETWORK_USER_4",       WIDTH_NETWORK_USER_4,       "Fig.4"),
        ("OP_USER_4",                  OP_USER_4,                  "Fig.4"),
        # Fig.5
        ("LENGTH_NETWORK_DT_5",        LENGTH_NETWORK_DT_5,        "Fig.5"),
        ("WIDTH_NETWORK_DT_5",         WIDTH_NETWORK_DT_5,         "Fig.5"),
        ("NUM_USERS_DT_5",             NUM_USERS_DT_5,             "Fig.5"),
        ("OP_DT_5",                    OP_DT_5,                    "Fig.5"),
    ]
    df = pd.DataFrame(rows, columns=["Parameter", "Value", "Used In"])
    df.to_excel(writer, sheet_name="Run_Params", index=False)


def run_and_get_metrics(params, user_sets_list):
    simulator = EventSimulator(
        length_network=params['length_network'],
        width_network=params['width_network'],
        edge_length_km=EDGE_LENGTH_KM,
        num_users=params['num_users'],
        p_op=params['p_op'],
        max_per_edge=MAX_PER_EDGE,
        decoherence_time=params.get('decoherence_time', DECOHERENCE_TIME),
        max_timeslot=MAX_TIMEESLOT_PER_TRIAL
    )

    dr_object = EntanglementDistribution()
    random.seed(RANDOM_SEED)

    deployed_dicts_per_trial = simulator.run_trials(
        user_sets=user_sets_list,
        routing_method=params['routing_method'],
        source_method=params['source_method'],
        seed=RANDOM_SEED,
        dr_object=dr_object,
        cost_budget=params['cost_budget']
    )

    summary = dr_object.get_summary_dict()
    summary['deployed_dicts'] = str(deployed_dicts_per_trial)

    return summary


def create_combo_plot(title, x_label='Quantum Source Budget'):
    """
    Creates a matplotlib figure and axes with a shared x-axis.
    Cost-Efficiency (CE) is a line plot on the left Y-axis.
    Distribution Rate (DR) is a bar plot on the right Y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Cost-Efficiency (CE)', color='tab:blue', fontsize=16)
    ax2.set_ylabel('Distribution Rate (DR)', color='tab:red', fontsize=16)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
    ax1.set_title(title, fontsize=18)

    return fig, ax1, ax2


def create_dr_plot(title, x_label='Quantum Source Budget'):
    """
    Creates a single-axis bar plot for Distribution Rate (DR) only.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel('Distribution Rate (DR)', fontsize=18)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    return fig, ax


def line_style_for(method: str) -> str:
    return '-' if method == 'steiner_tree' else '--'


def bar_style_kwargs(method: str, color):
    if method == 'steiner_tree':
        return dict(color=color, alpha=0.7)
    else:
        return dict(facecolor='none', edgecolor=color, hatch='//', linewidth=1.5)


def place_legend_inside(ax, loc='upper left'):
    return ax.legend(
        loc=loc,
        bbox_to_anchor=(0.02, 0.98),  # 轴内偏移（左上角附近）
        borderaxespad=0.4, frameon=True, framealpha=0.9,
        prop={'size': 16}
    )


def _plot_dr(ax, results_dict, cost_budgets, group_key_fn):
    """
    仅绘制 DR 的柱状图。
    results_dict: {label: {'ce_actual': [...], 'dr': [...]}}, 仍可保留 ce_actual 以便导出 Excel，但不绘图。
      label 例如: 'SP_steiner_tree' / 'SP_all_edges'
    group_key_fn(label) -> 用于分组上色的 key（同色条件）
    """
    base_x = np.array(list(cost_budgets), dtype=float)
    groups = []
    for label in results_dict.keys():
        g = group_key_fn(label)
        if g not in groups:
            groups.append(g)
    group_to_color_idx = {g: i for i, g in enumerate(groups)}

    num_series = len(results_dict)
    step = base_x[1] - base_x[0] if len(base_x) > 1 else 1.0
    bar_width = step / (num_series + 1)

    # 稳定顺序绘制：颜色按组一致；steiner=实心，all_edges=空心+斜纹
    for i, (label, data) in enumerate(results_dict.items()):
        try:
            _key, method = label.split('-', 1)
        except ValueError:
            method = 'steiner_tree'  # 兜底

        color = plt.cm.get_cmap('tab10')(group_to_color_idx[group_key_fn(label)] % 10)

        if method == 'steiner_tree':
            bar_kwargs = dict(color=color, alpha=0.8)
        else:
            bar_kwargs = dict(facecolor='none', edgecolor=color, hatch='//', linewidth=1.5)

        x = base_x + (i - (num_series - 1) / 2) * bar_width
        ax.bar(x, data['dr'], width=bar_width, label=label, **bar_kwargs)

    ax.set_xticks(base_x)
    ax.set_xticklabels(cost_budgets)


def _plot_combo(ax1, ax2, results_dict, cost_budgets, group_key_fn):
    """
    results_dict: {label: {'ce_actual': [...], 'dr': [...]}}
      其中 label: 'SP_steiner_tree' / 'SP_all_edges'
    group_key_fn(label) -> 用于分组上色的 key（同色条件）
    """
    # 统一 x
    base_x = np.array(list(cost_budgets), dtype=float)
    groups = []
    for label in results_dict.keys():
        g = group_key_fn(label)
        if g not in groups:
            groups.append(g)
    group_to_color_idx = {g: i for i, g in enumerate(groups)}

    num_series = len(results_dict)
    step = base_x[1] - base_x[0] if len(base_x) > 1 else 1.0
    bar_width = step / (num_series + 1)

    # 用稳定顺序绘制
    for i, (label, data) in enumerate(results_dict.items()):
        # 解析 method
        try:
            _key, method = label.split('_', 1)
        except ValueError:
            method = 'steiner_tree'  # 兜底
        # 同色：按组 key 选颜色
        color = plt.cm.get_cmap('tab10')(group_to_color_idx[group_key_fn(label)] % 10)

        ax1.plot(
            base_x, data['ce_actual'],
            marker='o', linestyle=line_style_for(method), color=color, markersize=5,
            label=f'{label} (CE)'
        )

        # 柱子（DR）：同色，steiner=实心，all=hatch
        x = base_x + (i - (num_series - 1) / 2) * bar_width
        ax2.bar(x, data['dr'], width=bar_width, label=f'{label} (DR)', **bar_style_kwargs(method, color))

    ax2.set_xticks(base_x)
    ax2.set_xticklabels(cost_budgets)


def plot_protocols_vs_budget(excel_writer, output_dir, user_sets_list):
    """Generates Plot 1: Protocols vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 1: Protocols vs. Budget   ###")
    print("#" * 60)

    protocols = ['Reactive Routing', 'Proactive Routing']
    source_methods = ["all_edges", "steiner_tree"]
    cost_budgets = COST_BUDGETS

    all_plot_data = []
    results = {f'{p}-{sm}': {'ce_actual': [], 'dr': []} for p in protocols for sm in source_methods}

    for method in source_methods:
        for protocol in protocols:
            for budget in cost_budgets:
                label = f'{protocol}-{method}'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': LENGTH_NETWORK_PROTOCOLS_1, 'width_network': WIDTH_NETWORK_PROTOCOLS_1,
                    'num_users': NUM_USERS_PROTOCOLS_1,
                    'p_op': OP_PROTOCOLS_1, 'routing_method': protocol,
                    'source_method': method, 'cost_budget': budget
                }
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {
                    'Protocol': protocol,
                    'Source Method': method,
                    'Budget': budget,
                    'Deployed_Dicts': summary_data.get('deployed_dicts', 'N/A'),
                    'user_sets': str(user_sets_list)
                }
                plot_point_data.update(summary_data)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='Protocols_vs_Budget', index=False)

    fig, ax = create_dr_plot('Protocols vs. Quantum Source Budget: DR')
    _plot_dr(
        ax, results, cost_budgets,
        group_key_fn=lambda lbl: lbl.split('-', 1)[0]
    )

    place_legend_inside(ax, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_protocols_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 1 saved and data exported to Excel.")


def plot_mpp_op_vs_budget(excel_writer, output_dir, user_sets_list):
    """Generates Plot 2: MP-P p_op vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 2: MP-P p_op vs. Budget   ###")
    print("#" * 60)

    p_ops = [0.6, 0.7, 0.8, 0.9]
    source_methods = ["all_edges", "steiner_tree"]
    cost_budgets = COST_BUDGETS

    all_plot_data = []
    results = {f'p_op{p_op}-{sm}': {'ce_actual': [], 'dr': []} for p_op in p_ops for sm in source_methods}

    for method in source_methods:
        for p_op in p_ops:
            for budget in cost_budgets:
                label = f'p_op{p_op}-{method}'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': LENGTH_NETWORK_OP_2, 'width_network': WIDTH_NETWORK_OP_2,
                    'num_users': NUM_USERS_OP_2,
                    'p_op': p_op, 'routing_method': 'Proactive Routing',
                    'source_method': method, 'cost_budget': budget
                }
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {
                    'p_op': p_op,
                    'Source Method': method,
                    'Budget': budget,
                    'Deployed_Dicts': summary_data.get('deployed_dicts', 'N/A'),
                    'user_sets': str(user_sets_list)
                }
                plot_point_data.update(summary_data)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='Proactive Routing Operation Probability', index=False)

    fig, ax = create_dr_plot('Proactive Routing Operation Probability vs. Budget: DR')
    _plot_dr(
        ax, results, cost_budgets,
        group_key_fn=lambda lbl: lbl.split('-', 1)[0]
    )

    place_legend_inside(ax, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '2_mpp_operation_probability_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 2 saved and data exported to Excel.")


def plot_mpp_scalability_network_size_vs_budget(excel_writer, output_dir, base_user_sets_lists):
    """Generates Plot 3: MP-P Scalability (Network Size) vs. Budget (Refactored)"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 3: MP-P Scalability (Network Size) vs. Budget   ###")
    print("#" * 60)

    network_sizes = [(3, 3), (4, 4), (5, 5)]
    source_methods = ["all_edges", "steiner_tree"]
    cost_budgets = COST_BUDGETS

    all_plot_data = []
    results = {f'{s[0]}x{s[1]}-{sm}': {'ce_actual': [], 'dr': []} for s in network_sizes for sm in source_methods}

    for method in source_methods:
        for size in network_sizes:
            user_sets_list = base_user_sets_lists[f'{size[0]}x{size[1]}']
            for budget in cost_budgets:
                label = f'{size[0]}x{size[1]}-{method}'
                print(f"\n--- Running: {size[0]}x{size[1]} ({method}), Budget={budget} ---")
                params = {
                    'length_network': size[0], 'width_network': size[1], 'num_users': NUM_USERS_NET_SIZE_3,
                    'p_op': OP_NET_SIZE_3, 'routing_method': 'Proactive Routing',
                    'source_method': method, 'cost_budget': budget
                }
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {
                    'Network_Size': f'{size[0]}x{size[1]}',
                    'Source Method': method,
                    'Budget': budget,
                    'Deployed_Dicts': summary_data.get('deployed_dicts', 'N/A'),
                    'user_sets': str(user_sets_list)
                }
                plot_point_data.update(summary_data)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='Proactive Routing_Scalability_Network', index=False)

    fig, ax = create_dr_plot('Proactive Routing Scalability (Network Size) vs. Budget: DR')
    _plot_dr(
        ax, results, cost_budgets,
        group_key_fn=lambda lbl: lbl.split('-', 1)[0]
    )

    place_legend_inside(ax, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '3_mpp_scalability_network_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 3 saved and data exported to Excel.")


def plot_mpp_scalability_users_vs_budget(excel_writer, output_dir, base_user_sets_lists):
    """Generates Plot 4: MP-P Scalability (Number of Users) vs. Budget (Refactored)"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT 4: MPP Scalability (Number of Users) vs. Budget   ###")
    print("#" * 60)

    num_users_list = [3, 4, 5]
    source_methods = ["all_edges", "steiner_tree"]
    cost_budgets = COST_BUDGETS

    all_plot_data = []
    results = {f'{n}users-{sm}': {'ce_actual': [], 'dr': []} for n in num_users_list for sm in source_methods}

    for method in source_methods:
        for num_users in num_users_list:
            user_sets_list = base_user_sets_lists[f'{num_users}_users']
            for budget in cost_budgets:
                label = f'{num_users}users-{method}'
                print(f"\n--- Running: {num_users} Users ({method}), Budget={budget} ---")
                params = {
                    'length_network': LENGTH_NETWORK_USER_4, 'width_network': WIDTH_NETWORK_USER_4, 'num_users': num_users,
                    'p_op': OP_USER_4, 'routing_method': 'Proactive Routing',
                    'source_method': method, 'cost_budget': budget
                }
                summary_data = run_and_get_metrics(params, user_sets_list)
                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {
                    'Num_Users': num_users,
                    'Source Method': method,
                    'Budget': budget,
                    'Deployed_Dicts': summary_data.get('deployed_dicts', 'N/A'),
                    'user_sets': str(user_sets_list)
                }
                plot_point_data.update(summary_data)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='Proactive Routing_Scalability_Users', index=False)

    fig, ax = create_dr_plot('Proactive Routing Scalability (Number of Users) vs. Budget: DR')
    _plot_dr(
        ax, results, cost_budgets,
        group_key_fn=lambda lbl: lbl.split('-', 1)[0]
    )
    place_legend_inside(ax, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_mpp_scalability_users_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 4 saved and data exported to Excel.")


def plot_mpp_decoherence_vs_budget(excel_writer, output_dir, user_sets_list):
    """Generates Plot: MP-P Decoherence Time vs. Budget"""
    print("\n" + "#" * 60)
    print("###   GENERATING PLOT: MP-P Decoherence Time vs. Budget   ###")
    print("#" * 60)

    decoherence_times = [1, 2, 3, 4]  # Varying decoherence time
    source_methods = ["all_edges", "steiner_tree"]
    cost_budgets = COST_BUDGETS

    all_plot_data = []
    results = {f'dt{dt}-{sm}': {'ce_actual': [], 'dr': []} for dt in decoherence_times for sm in source_methods}

    for method in source_methods:
        for dt in decoherence_times:
            for budget in cost_budgets:
                label = f'dt{dt}-{method}'
                print(f"\n--- Running: {label}, Budget={budget} ---")
                params = {
                    'length_network': LENGTH_NETWORK_DT_5, 'width_network': WIDTH_NETWORK_DT_5,
                    'num_users': NUM_USERS_DT_5,
                    'p_op': OP_DT_5,
                    'routing_method': 'Proactive Routing',
                    'source_method': method,
                    'cost_budget': budget,
                    'decoherence_time': dt  # Use the varying decoherence time
                }
                summary_data = run_and_get_metrics(params, user_sets_list)

                results[label]['ce_actual'].append(summary_data['cost_efficiency_actual'])
                results[label]['dr'].append(summary_data['average_dr'])

                plot_point_data = {
                    'Decoherence_Time': dt,
                    'Source Method': method,
                    'Budget': budget,
                    'Deployed_Dicts': summary_data.get('deployed_dicts', 'N/A'),
                    'user_sets': str(user_sets_list)
                }
                plot_point_data.update(summary_data)
                all_plot_data.append(plot_point_data)

    df = pd.DataFrame(all_plot_data)
    df.to_excel(excel_writer, sheet_name='Proactive Routing_Decoherence_Time', index=False)

    fig, ax = create_dr_plot('Proactive Routing Decoherence Time vs. Budget: DR')
    _plot_dr(
        ax, results, cost_budgets,
        group_key_fn=lambda lbl: lbl.split('-', 1)[0]  # Group by 'dt1', 'dt2', etc.
    )

    place_legend_inside(ax, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_mpp_decoherence_vs_budget.png'))
    plt.close(fig)
    print("\n[SUCCESS] Plot 5 on Decoherence Time saved and data exported to Excel.")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_directory = os.path.join('simulation_plots', timestamp)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Created output directory: {output_directory}")

    excel_filepath = os.path.join(output_directory, f"simulation_results_{timestamp}.xlsx")

    print("Pre-generating user sets for consistency...")
    random.seed(RANDOM_SEED)

    sim_for_users_p1_p2_p5 = EventSimulator(LENGTH_NETWORK_PROTOCOLS_1, WIDTH_NETWORK_PROTOCOLS_1, EDGE_LENGTH_KM,
                                            MAX_PER_EDGE, OP_PROTOCOLS_1, DECOHERENCE_TIME, NUM_USERS_PROTOCOLS_1)
    user_sets_p1_p2_p5 = [sim_for_users_p1_p2_p5.user_gen.random_users(NUM_USERS_PROTOCOLS_1) for _ in range(NUM_TRIALS)]

    base_user_sets_p3 = {}
    for size in [(3, 3), (4, 4), (5, 5)]:
        sim = EventSimulator(size[0], size[1], EDGE_LENGTH_KM, MAX_PER_EDGE, OP_NET_SIZE_3, DECOHERENCE_TIME, NUM_USERS_NET_SIZE_3)
        base_user_sets_p3[f'{size[0]}x{size[1]}'] = [sim.user_gen.random_users(NUM_USERS_NET_SIZE_3) for _ in range(NUM_TRIALS)]

    base_user_sets_p4 = {}
    sim_for_p4 = EventSimulator(LENGTH_NETWORK_USER_4, WIDTH_NETWORK_USER_4, EDGE_LENGTH_KM, MAX_PER_EDGE, OP_USER_4,
                                DECOHERENCE_TIME, num_users=5)  # Max users
    for num_users in [3, 4, 5]:
        base_user_sets_p4[f'{num_users}_users'] = [sim_for_p4.user_gen.random_users(k=num_users) for _ in range(NUM_TRIALS)]

    print("User set generation complete.")

    with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
        pd.DataFrame([{"status": "initializing"}]).to_excel(writer, sheet_name="README", index=False)

        export_run_parameters(writer)

        plot_protocols_vs_budget(writer, output_directory, user_sets_p1_p2_p5)
        plot_mpp_op_vs_budget(writer, output_directory, user_sets_p1_p2_p5)
        plot_mpp_decoherence_vs_budget(writer, output_directory, user_sets_p1_p2_p5)

        # plot_mpp_scalability_network_size_vs_budget(writer, output_directory, base_user_sets_p3)
        # plot_mpp_scalability_users_vs_budget(writer, output_directory, base_user_sets_p4)

    print("\n\n" + "*" * 50)
    print("********** ALL SIMULATIONS COMPLETE   **********")
    print(f"All plot images and data saved in '{output_directory}'.")
    print("*" * 50)
