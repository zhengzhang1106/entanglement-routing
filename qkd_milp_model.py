"""
MILP model for Satellite-HAP-GS QKD resource allocation.

First-version assumptions:
1. Time is slotted.
2. Link capacity input is already converted to key bits per slot.
3. GS, satellite, and HAP nodes are trusted QKD nodes.
4. QKP is link-based: each edge has its own stored key pool.
5. Primary objective: maximize served secret keys.
6. Secondary objective: maximize residual QKP bits at the final slot.
"""

from collections import defaultdict
import pulp


class QKDMILPModel:
    def __init__(self, qkd_network, demands, beta=1e-4, model_name="sat_hap_qkd_milp"):
        self.network = qkd_network
        self.topology = qkd_network.topology
        self.demands = demands
        self.beta = beta
        self.model_name = model_name

        self.model = None
        self.z = {}
        self.x = {}
        self.y = {}
        self.r = {}
        self.q = {}

    def _build_sets(self):
        self.T = sorted(set(self.network.time_slots) | {d.time_slot for d in self.demands})
        self.N = self.network.get_all_nodes()
        self.E = [tuple(sorted(e)) for e in self.network.get_all_edges()]

        # Directed arcs are used for key-flow conservation.
        self.A = []
        for u, v in self.E:
            self.A.append((u, v))
            self.A.append((v, u))

        self.D = self.demands
        self.demands_by_time = defaultdict(list)
        for d in self.D:
            self.demands_by_time[d.time_slot].append(d)

    def build_model(self):
        self._build_sets()
        model = pulp.LpProblem(self.model_name, pulp.LpMaximize)

        # z[t,e] = 1 if physical QKD link e is activated at time slot t.
        for t in self.T:
            for e in self.E:
                self.z[(t, e)] = pulp.LpVariable(
                    f"z_t{t}_{e[0]}_{e[1]}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpBinary,
                )

        # r[d] = served key bits for demand d.
        for d in self.D:
            self.r[d.demand_id] = pulp.LpVariable(
                f"r_{d.demand_id}",
                lowBound=0,
                upBound=d.key_bits,
                cat=pulp.LpContinuous,
            )

        # x = key bits consumed from newly generated link capacity.
        # y = key bits consumed from QKP virtual links.
        for d in self.D:
            t = d.time_slot
            for a in self.A:
                self.x[(d.demand_id, t, a)] = pulp.LpVariable(
                    f"x_{d.demand_id}_t{t}_{a[0]}_{a[1]}",
                    lowBound=0,
                    cat=pulp.LpContinuous,
                )
                self.y[(d.demand_id, t, a)] = pulp.LpVariable(
                    f"y_{d.demand_id}_t{t}_{a[0]}_{a[1]}",
                    lowBound=0,
                    cat=pulp.LpContinuous,
                )

        # q[t,e] = QKP stored key bits on edge e at the end of time slot t.
        for t in self.T:
            for e in self.E:
                self.q[(t, e)] = pulp.LpVariable(
                    f"q_t{t}_{e[0]}_{e[1]}",
                    lowBound=0,
                    cat=pulp.LpContinuous,
                )

        self._add_flow_conservation_constraints(model)
        self._add_link_capacity_constraints(model)
        self._add_qkp_constraints(model)
        self._add_connection_limit_constraints(model)
        self._add_objective(model)

        self.model = model
        return model

    def _add_flow_conservation_constraints(self, model):
        for d in self.D:
            t = d.time_slot

            for n in self.N:
                outgoing = []
                incoming = []

                for u, v in self.A:
                    if u == n:
                        outgoing.append(self.x[(d.demand_id, t, (u, v))])
                        outgoing.append(self.y[(d.demand_id, t, (u, v))])
                    if v == n:
                        incoming.append(self.x[(d.demand_id, t, (u, v))])
                        incoming.append(self.y[(d.demand_id, t, (u, v))])

                lhs = pulp.lpSum(outgoing) - pulp.lpSum(incoming)

                if n == d.src:
                    model += lhs == self.r[d.demand_id], f"flow_src_{d.demand_id}_{n}"
                elif n == d.dst:
                    model += lhs == -self.r[d.demand_id], f"flow_dst_{d.demand_id}_{n}"
                else:
                    model += lhs == 0, f"flow_mid_{d.demand_id}_{n}"

    def _add_link_capacity_constraints(self, model):
        for t in self.T:
            for e in self.E:
                u, v = e
                capacity = self.network.get_capacity(t, u, v)
                used_new_keys = []

                for d in self.demands_by_time.get(t, []):
                    used_new_keys.append(self.x[(d.demand_id, t, (u, v))])
                    used_new_keys.append(self.x[(d.demand_id, t, (v, u))])

                model += (
                    pulp.lpSum(used_new_keys) <= capacity * self.z[(t, e)]
                ), f"capacity_t{t}_{u}_{v}"

    def _add_qkp_constraints(self, model):
        for idx, t in enumerate(self.T):
            prev_t = self.T[idx - 1] if idx > 0 else None

            for e in self.E:
                u, v = e
                capacity = self.network.get_capacity(t, u, v)
                used_new = []
                used_qkp = []

                for d in self.demands_by_time.get(t, []):
                    used_new.append(self.x[(d.demand_id, t, (u, v))])
                    used_new.append(self.x[(d.demand_id, t, (v, u))])
                    used_qkp.append(self.y[(d.demand_id, t, (u, v))])
                    used_qkp.append(self.y[(d.demand_id, t, (v, u))])

                if prev_t is None:
                    q_prev_expr = self.network.get_qkp(u, v)
                else:
                    q_prev_expr = self.q[(prev_t, e)]

                # QKP causality: stored keys consumed in slot t must already exist.
                model += (
                    pulp.lpSum(used_qkp) <= q_prev_expr
                ), f"qkp_causality_t{t}_{u}_{v}"

                # End-of-slot QKP update.
                model += (
                    self.q[(t, e)]
                    == q_prev_expr
                    + capacity * self.z[(t, e)]
                    - pulp.lpSum(used_new)
                    - pulp.lpSum(used_qkp)
                ), f"qkp_update_t{t}_{u}_{v}"

    def _add_connection_limit_constraints(self, model):
        for t in self.T:
            for n in self.N:
                limit = self.topology.get_connection_limit(n)
                incident_z = []

                for e in self.E:
                    if n in e:
                        incident_z.append(self.z[(t, e)])

                model += (
                    pulp.lpSum(incident_z) <= limit
                ), f"connection_limit_t{t}_{n}"

    def _add_objective(self, model):
        served_keys = pulp.lpSum(self.r[d.demand_id] for d in self.D)
        final_t = max(self.T)
        final_qkp = pulp.lpSum(self.q[(final_t, e)] for e in self.E)
        model += served_keys + self.beta * final_qkp

    def solve(self, msg=True, time_limit=None):
        if self.model is None:
            self.build_model()

        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
        self.model.solve(solver)
        return self.get_solution()

    def get_solution(self):
        status = pulp.LpStatus[self.model.status]

        served = {
            d.demand_id: pulp.value(self.r[d.demand_id])
            for d in self.D
        }

        activated_links = {}
        for (t, e), var in self.z.items():
            val = pulp.value(var)
            if val is not None and val > 0.5:
                activated_links.setdefault(t, []).append(e)

        final_qkp = {}
        if self.T:
            final_t = max(self.T)
            for e in self.E:
                final_qkp[e] = pulp.value(self.q[(final_t, e)])

        total_served = sum(v for v in served.values() if v is not None)

        return {
            "status": status,
            "total_served_keys": total_served,
            "served_by_demand": served,
            "activated_links": activated_links,
            "final_qkp": final_qkp,
            "objective": pulp.value(self.model.objective),
        }
