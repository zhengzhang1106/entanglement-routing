"""
Manage quantum memory with limited number of slots for each node.
Automatically purges expired entanglement links that exceed the decoherence time.

Each entanglement record includes:
    - peer_id: the id of remoted node that build the entanglement link together
    - gen_time: time at which the entanglement was generated
    - fidelity: the current quality of the entangled state (optional)

Memory capacity:
    - size: number of memory slots
    - decoherence_time: max valid lifetime (in time slots)

Key Methods:
    - occupy_memory(peer_id, gen_time, fidelity=1.0)
    - release_memory(current_time)
    - show_memory(current_time)
    - compute_fidelity(gen_time, current_time)

Note:
    After decoherence time, the qubit is assumed to have undergone decoherence and is discarded
    Fidelity decay (exponential or linear) is optional and currently disabled.
"""


import math


class QuantumMemory:
    def __init__(self, node_id, size=4, decoherence_time=10, fidelity_decay=None, decay_rate=0.01):
        self.node_id = node_id
        self.size = size
        self.decoherence_time = decoherence_time
        self.fidelity_decay = fidelity_decay
        self.decay_rate = decay_rate
        self.memory_storage = {}  # key: (peer_id, gen_time), value: fidelity

    def occupy_memory(self, peer_id, gen_time, fidelity=1.0):
        if len(self.memory_storage) >= self.size:
            print("Not enough memory!")
            return False
        self.memory_storage[(peer_id, gen_time)] = fidelity
        return True

    def release_memory(self, current_time):
        self.memory_storage = {
            (peer_id, gen_time): f
            for (peer_id, gen_time), f in self.memory_storage.items()
            if current_time - gen_time < self.decoherence_time
        }

    def show_memory(self, current_time):
        self.release_memory(current_time)
        print(f"  Memory Size: {self.size}, Memory Usage: {len(self.memory_storage)}")
        for (peer_id, gen_time), _ in self.memory_storage.items():
            f = self.compute_fidelity(gen_time, current_time)
            print(f"  Peer_id: {peer_id}, Gen_time: {gen_time}, Fidelity: {f:.2f}")

    def compute_fidelity(self, gen_time, current_time):
        if self.fidelity_decay is None:
            return 1.0
        dt = current_time - gen_time
        if self.fidelity_decay == 'exponential':
            return math.exp(-self.decay_rate * dt)
        elif self.fidelity_decay == 'linear':
            return max(0.0, 1.0 - self.decay_rate * dt)


if __name__ == "__main__":
    memory = QuantumMemory(node_id="A", size=3, decoherence_time=8, decay_rate=0.05)

    memory.occupy_memory("B", 0)
    memory.occupy_memory("C", 5)
    memory.occupy_memory("D", 7)
    memory.show_memory(current_time=7)
    memory.show_memory(current_time=9)