import random


class RequestGenerator:
    def __init__(self, all_nodes):
        self.all_nodes = all_nodes

    def random_users(self, k):
        """
        Randomly select k user nodes from the topology.
        """
        if k > len(self.all_nodes):
            raise ValueError("Number of users exceeds available nodes.")
        return random.sample(self.all_nodes, k)
