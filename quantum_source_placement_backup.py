import math
import random
import networkx as nx
from collections import defaultdict
from steiner_tree_algorithms import approximate_steiner_tree

# ===== 全局成本：每对 source 的代价 =====
PAIR_COST = 2  # 需要改成本时，在此修改即可


class SourcePlacement:
    """
    SRP-MT: Source-Redundant Provisioning over Multiple Trees
    - 多棵候选 Steiner 树的并集上，基于全局预算进行冗余部署（离散贪心）
    - 目标近似：最大化 sum_k w_k * sum_{e in T_k} log(1 - (1 - p_e)^{x_e})
    - 约束：sum_e x_e <= B, 0 <= x_e <= max_per_edge, （可选）节点内存
    """

    def __init__(self, topo):
        self.topo = topo
        self.sources = []  # list of (u, v) for each placed pair

    def place_sources_for_request(
        self,
        user_set,
        method="OP",
        cost_budget=None,
        max_per_edge=1,
        # --- SRP-MT 参数 ---
        k_trees=5,
        tree_weights=None,
        # --- 物理参数（用于 p_e 计算）---
        p_op=1.0,                    # 单次操作成功率
        loss_coef_dB_per_km=0.2,     # 衰减系数（dB/km）
        seed=1,
        node_memory=None
    ):
        """
        Returns:
            list[(u,v)]: 每个元组是一对 source 的部署位置（同一边会重复出现）
        """
        random.seed(seed)

        # 兼容原基线方法
        if method in ("OP", "steiner_tree"):
            return self._baseline_round_robin(user_set, base="steiner",
                                              cost_budget=cost_budget, max_per_edge=max_per_edge)
        if method in ("NOP", "all_edges"):
            return self._baseline_round_robin(user_set, base="all",
                                              cost_budget=cost_budget, max_per_edge=max_per_edge)

        if method.lower() != "srp_mt":
            raise ValueError(f"Unknown source placement method: {method}")

        # === SRP-MT ===
        if cost_budget is None or cost_budget < 0:
            raise ValueError("[SRP-MT] cost_budget 必须给定且非负")
        if cost_budget % 1 != 0:
            print(f"[SRP-MT][WARN] cost_budget={cost_budget} 不是整数对，按 {int(cost_budget)} 使用")
        budget_pairs = int(cost_budget)

        # 1) 生成候选树
        trees = self._gen_multi_steiner_trees(user_set, k_trees=k_trees, seed=seed)
        if not trees:
            self.sources = []
            print("[SRP-MT] 未生成候选树，停止。")
            return self.sources

        # 2) 计算每条边的 p_e（用给定物理公式）
        p_e = self._get_edge_success_prob(p_op=p_op, loss_coef_dB_per_km=loss_coef_dB_per_km)

        # 3) 多树边并集 & 权重
        E_union = sorted({self._norm_edge(e) for T in trees for e in T.edges()})
        if not E_union:
            self.sources = []
            print("[SRP-MT] 候选树无边。")
            return self.sources

        if tree_weights is None:
            tree_weights = [1.0] * len(trees)
        assert len(tree_weights) == len(trees), "tree_weights 长度需与 k_trees 一致"

        # 4) 贪心渐进取整
        x = {e: 0 for e in E_union}
        used = 0

        mem_left = None
        if node_memory is not None:
            mem_left = dict(node_memory)

        # Δ_e(x) = sum_k w_k [ log(1-(1-p_e)^{x+1}) - log(1-(1-p_e)^{x}) ]，若 e∉T_k 则该项为0
        def delta_mt(edge, x_e):
            p = p_e.get(edge, 0.0)
            if p <= 0.0:
                return -1e18
            base = (1 - p) ** x_e
            nxt = (1 - p) ** (x_e + 1)
            if base >= 1.0:
                return -1e18
            gain_single = math.log(1 - nxt) - math.log(1 - base)

            g = 0.0
            edge_sets = self._tree_edge_sets_cache(trees)
            for wk, Ek in zip(tree_weights, edge_sets):
                if edge in Ek:
                    g += wk * gain_single
            return g

        # 预缓存每棵树的边集（加速）
        _ = self._tree_edge_sets_cache(trees)

        while used < budget_pairs:
            best_edge, best_gain = None, -1e18
            for e in E_union:
                if x[e] >= max_per_edge:
                    continue
                if not self._node_feasible_after_plus1(e, x, mem_left):
                    continue
                g = delta_mt(e, x[e])
                if g > best_gain:
                    best_gain, best_edge = g, e

            if best_edge is None:
                break

            x[best_edge] += 1
            used += 1
            if mem_left is not None:
                u, v = best_edge
                mem_left[u] -= 1
                mem_left[v] -= 1

        # 5) 展开 sources
        self.sources = []
        for e, cnt in x.items():
            self.sources.extend([e] * cnt)

        capacity_pairs = len(E_union) * max_per_edge
        print(f"[SRP-MT] trees={len(trees)}, union_edges={len(E_union)}")
        print(f"[SRP-MT] placed_pairs={used}, budget_pairs={budget_pairs}, capacity_pairs={capacity_pairs}")
        print(f"[SRP-MT] max_per_edge={max_per_edge}, node_memory={'ON' if node_memory else 'OFF'}")
        print(f"[SourcePlacement] Total cost: {self.compute_cost()} (PAIR_COST={PAIR_COST})")
        return self.sources

    # ===== 基线方法（保持原 OP/NOP 逻辑，预算与成本改用 PAIR_COST） =====
    def _baseline_round_robin(self, user_set, base="steiner", cost_budget=None, max_per_edge=1):
        # 1) 选基线边集
        if base == "steiner":
            subgraph = approximate_steiner_tree(self.topo.graph, user_set)
            base_edges = list(subgraph.edges())
        else:
            base_edges = list(self.topo.get_edges())

        base_keys = sorted({self._norm_edge(e) for e in base_edges})
        if not base_keys:
            self.sources = []
            print("[SourcePlacement] No candidate edges found.")
            return self.sources

        self.sources = []
        per_edge_count = {k: 0 for k in base_keys}

        # 2) 无预算：每边放 1（不超过上限）
        if cost_budget is None:
            for (u, v) in base_keys:
                if per_edge_count[(u, v)] < max_per_edge:
                    self.sources.append((u, v))
                    per_edge_count[(u, v)] += 1
            print(f"[SourcePlacement] Method: {base}, Sources placed: {self.sources}")
            print(f"[SourcePlacement] Total cost: {self.compute_cost()}")
            print("[SourcePlacement] Cost budget: None")
            return self.sources

        if cost_budget < 0:
            raise ValueError("cost_budget must be non-negative")
        if cost_budget % 1 != 0:
            print(f"[SourcePlacement][WARN] cost_budget={cost_budget} 非整数对, 使用 {int(cost_budget)}。")
        budget_pairs = int(cost_budget)

        capacity_pairs = len(base_keys) * max_per_edge
        target_pairs = min(budget_pairs, capacity_pairs)

        # 4) 轮转分配
        placed = 0
        idx = 0
        n = len(base_keys)
        while placed < target_pairs:
            u, v = base_keys[idx % n]
            if per_edge_count[(u, v)] < max_per_edge:
                self.sources.append((u, v))
                per_edge_count[(u, v)] += 1
                placed += 1
            idx += 1
            if idx >= n and all(per_edge_count[k] >= max_per_edge for k in base_keys):
                break  # capacity exhausted

        # 5) 输出
        print(f"[SourcePlacement] Method: {base}, Sources placed: {self.sources}")
        print(f"[SourcePlacement] Total cost: {self.compute_cost()} (target_pairs={target_pairs})")
        print(f"[SourcePlacement] Cost budget (pairs): {cost_budget}, max_per_edge={max_per_edge}, "
              f"capacity_pairs={capacity_pairs}, used_pairs={placed}")
        return self.sources

    # ===== 工具 =====
    def compute_cost(self):
        # 使用全局 PAIR_COST，不再写死 2
        return len(self.sources) * PAIR_COST

    @staticmethod
    def _norm_edge(e):
        u, v = e[:2]
        return (u, v) if u < v else (v, u)

    def _get_edge_success_prob(self, p_op, loss_coef_dB_per_km):
        """
        使用给定物理模型：
            p_loss = 1 - 10^(-(loss_coef_dB_per_km * L)/10)
            p_e = p_op * (1 - p_loss)
        其中 L 优先取边属性 'length_km'；若无则用 'weight'；再无则默认 1.0 km。
        """
        pe = {}
        G = self.topo.graph
        for (u, v, data) in G.edges(data=True):
            key = self._norm_edge((u, v))
            L = float(data.get("length_km", data.get("weight", 1.0)))
            p_loss = 1.0 - (10.0 ** (-(loss_coef_dB_per_km * L) / 10.0))
            p = p_op * (1.0 - p_loss)
            # 数值截断
            pe[key] = max(0.0, min(1.0, p))
        return pe

    def _node_feasible_after_plus1(self, e, x, mem_left):
        if mem_left is None:
            return True
        u, v = e
        return (mem_left.get(u, 1 << 30) >= 1) and (mem_left.get(v, 1 << 30) >= 1)

    # 缓存每棵树的“规范化边集”，用于快速增益计算
    def _tree_edge_sets_cache(self, trees):
        if not hasattr(self, "_edge_sets_cache"):
            self._edge_sets_cache = [set(self._norm_edge(e) for e in T.edges()) for T in trees]
        return self._edge_sets_cache

    def _gen_multi_steiner_trees(self, user_set, k_trees=5, seed=1):
        """
        通过对边权做轻微扰动生成 k 棵多样化的 Steiner 树
        """
        rng = random.Random(seed)
        trees = []
        G_orig = self.topo.graph

        for _ in range(k_trees):
            G = nx.Graph()
            for u, v, data in G_orig.edges(data=True):
                w = float(data.get("weight", 1.0))
                jitter = 1.0 + rng.uniform(-0.05, 0.05)
                G.add_edge(u, v, weight=w * jitter, length_km=data.get("length_km", w * jitter))
            try:
                T = approximate_steiner_tree(G, user_set)
                if len(T.edges()) > 0:
                    trees.append(self._to_undirected_simple(T))
            except Exception:
                continue

        # 去重
        uniq = {}
        for T in trees:
            key = tuple(sorted({self._norm_edge(e) for e in T.edges()}))
            uniq[key] = T
        # 清掉旧缓存
        if hasattr(self, "_edge_sets_cache"):
            delattr(self, "_edge_sets_cache")
        return list(uniq.values())

    @staticmethod
    def _to_undirected_simple(G):
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges(data=True))
        return H


if __name__ == "__main__":
    from network_topology import Topology

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
        (4, 7, 10),
        (5, 8, 10),
        (6, 7, 10),
        (7, 8, 10)
    ]
    topo = Topology(edge_list)
    users = [0, 2, 7]

    print("\n" + "=" * 50 + " SRP-MT " + "=" * 50 + "\n")
    sp = SourcePlacement(topo)
    placed = sp.place_sources_for_request(
        users,
        method="srp_mt",
        cost_budget=10,           # 以“对”为单位；总cost=PAIR_COST*10
        max_per_edge=3,
        k_trees=6,
        p_op=0.9,                 # 例：单次操作成功率
        loss_coef_dB_per_km=0.2,  # 例：0.2 dB/km
        seed=42,
        node_memory=None
    )
    print("SRP-MT placed:", placed)
    print("Total cost:", sp.compute_cost(), "(PAIR_COST =", PAIR_COST, ")")
