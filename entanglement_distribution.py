"""
Compute the Distribution Rate (DR) of GHZ states over multiple Monte Carlo trials.
- DR = average number of GHZ states per timeslot
- DR = (1/M) * sum_i(1/T_i) where T_i is the number of timeslots to generate a GHZ
- If a run fails to generate GHZ in 5000 timeslots, it is discarded
- If >5% of runs fail, the whole DR value is discarded
"""


class EntanglementDistribution:
    def __init__(self):
        self.successful_trials = []     # successful trials: list of (time_slot, num_ghz)
        self.failed_trials_num = 0          # number of failed attempts
        self.cost_list = []

    def record_trial(self, time_to_success, cost, num_ghz=1):
        """Record the number of timeslots needed to generate GHZ in one simulation."""
        if time_to_success == 0:
            self.failed_trials_num += 1
        else:
            self.successful_trials.append((time_to_success, num_ghz))
            self.cost_list.append(cost)

    def failure_rate(self):
        total = len(self.successful_trials) + self.failed_trials_num
        return self.failed_trials_num / total if total else 0.0

    def average_dr(self):
        if not self.successful_trials:
            return 0.0
        return sum([num_ghz / t for t, num_ghz in self.successful_trials]) / \
               (len(self.successful_trials) + self.failed_trials_num)

    def average_cost(self):
        if not self.cost_list:
            return 0.0
        return sum([cost for cost in self.cost_list]) / len(self.cost_list)

    def cost_effiency(self):
        if not self.successful_trials and not self.cost_list:
            return 0.0
        cost_effiency = []
        for i in range(len(self.successful_trials)):
            cost_effiency.append(self.successful_trials[i][1] / self.successful_trials[i][0] / self.cost_list[i])
        return sum([effiency for effiency in cost_effiency]) / len(cost_effiency)

    def is_valid_result(self):
        return self.failure_rate() <= 0.05

    def summary(self):
        print(f"  List of successful time slot : {[t for t, _ in self.successful_trials]}")
        print(f"  List of successful ghz count : {[n for _, n in self.successful_trials]}")
        print(f"  List of successful cost :      {self.cost_list}")
        print(f"  Total Runs")
        print(f"  Successful Runs : {len(self.successful_trials)}")
        print(f"  Failed Runs     : {self.failed_trials_num}")
        print(f"  Failure Rate    : {self.failure_rate():.2%}")

        print(f"  DR : {self.average_dr():.6f}")
        print(f"  cost : {self.average_cost():.6f}")
        print(f"  cost_effiency : {self.cost_effiency():.6f}")

        if self.is_valid_result():
            print(f"  DR : {self.average_dr():.6f}")
            print(f"  cost : {self.average_cost():.6f}")

        else:
            print("  [!] Too many failures. Discard this datapoint.")


if __name__ == "__main__":
    import random

    distribution = EntanglementDistribution()
    for _ in range(50):
        t = random.choice([random.randint(1, 500)] * 95 + [600] * 5)  # ~5% failure
        distribution.record_trial(t, cost=5, num_ghz=1)

    distribution.summary()
