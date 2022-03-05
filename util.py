from experiments import infectious_disease as ide

import numpy as np
from tqdm import trange


def _get_sick_state(states):
    n = np.zeros(states.shape[1], dtype=int)
    for i in range(states.shape[1]):
        col = states[:, i]
        n[i] = 1 if np.count_nonzero(col == 1) > 0 else 0
    return n

def _process_metrics(i, metrics, results):
    states = np.array(results["states"])
    num_sickdays = _get_sick_state(states)
    total_sickdays = np.sum(num_sickdays)
    metrics["num_sickdays"] += num_sickdays
    metrics["total_sickdays"] += total_sickdays

def run_simulations(n_sims, exp):
    size = ide.GRAPHS[exp.graph_name].number_of_nodes()
    metrics = {
        "num_sickdays": np.zeros(size),
        "total_sickdays": 0
    }

    exp.scenario_builder()
    for i in trange(n_sims):
        exp.seed = i
        np.random.seed(i)
        r = exp.run()
        _process_metrics(i, metrics, r["metric_results"])
    return metrics    