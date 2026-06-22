# Satellite-HAP-GS QKD Framework

This branch adds a first code framework for a Satellite-HAP-Ground Station QKD resource-allocation project.

The structure intentionally follows the flat style of the existing `entanglement-routing` repository: each major component is implemented as a standalone Python module in the repository root.

## Files

- `qkd_entity.py`: basic dataclasses for GS, satellite, HAP, link capacity, and demand.
- `qkd_topology.py`: NetworkX-based topology abstraction.
- `qkd_input_loader.py`: CSV input loader.
- `qkd_network.py`: QKD network state and QKP manager.
- `qkd_milp_model.py`: first multi-timeslot MILP skeleton.
- `qkd_heuristic.py`: greedy link-selection baseline skeleton.
- `qkd_metrics.py`: experiment metrics recorder.
- `qkd_simulator.py`: simulator wrapper.
- `run_qkd_simulator.py`: main run script.
- `data/qkd/`: toy input data.

## First-version modeling assumptions

1. Time is slotted.
2. Link capacity is already converted into key bits per time slot.
3. GS, satellite, and HAP nodes are trusted QKD nodes.
4. QKP is link-based, i.e., every physical link has its own key pool.
5. The primary objective is to maximize total served keys.
6. The secondary objective is to maximize residual QKP at the final time slot.
7. HAP count is treated as an input parameter, not a deployment decision.

## Run

```bash
pip install -r requirements_qkd.txt
python run_qkd_simulator.py
```

## Suggested next steps

1. Replace toy `data/qkd/link_capacities.csv` with SatQuMA2-generated satellite-GS and HAP-GS capacities.
2. Create separate input files for:
   - satellite-only
   - HAP-only
   - satellite + HAP
3. Validate MILP output on a small manually checked topology.
4. Add scenario comparison scripts and result plots.
5. Add richer constraints only after the base model is stable:
   - HAP mobility
   - satellite-HAP visibility
   - stronger cycle-prevention constraints
   - heuristic link assignment and GA-CKR-style routing
