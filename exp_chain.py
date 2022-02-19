import os
import sys
module_path = os.path.abspath(os.path.join('./ml-fairness-gym'))
if module_path not in sys.path:
    sys.path.append(module_path)

import networkx as nx
import numpy as np
from tqdm import trange
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use(["science", "notebook"])

from experiments import infectious_disease as ide
from agents import infectious_disease_agents as ida

def _num_sickdays(states):
    n = np.zeros(states.shape[1], dtype=int)
    for i in range(states.shape[1]):
        col = states[:, i]
        n[i] = np.count_nonzero(col == 1)
    return n

def process_metrics(i, metrics, results):
    states = np.array(results["states"])
    num_sickdays = _num_sickdays(states)
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
        r = exp.run()
        process_metrics(i, metrics, r["metric_results"])
    return metrics

if __name__ == "__main__":
    ide.GRAPHS["chain"] = nx.generators.path_graph(11)
    EXP = ide.Experiment()
    EXP.graph_name = "chain"
    EXP.infection_probability = 0.75
    EXP.infected_exit_probability = 1.0
    EXP.num_treatments = 1
    EXP.burn_in = 0
    
    EXP.num_steps = 10
    N = 1000

    EXP.agent_constructor = ide.NullAgent
    null_metrics = run_simulations(N, EXP)

    EXP.agent_constructor = ide.RandomAgent
    rng_metrics = run_simulations(N, EXP)

    EXP.agent_constructor = ida.MiddleAgent
    mid_metrics = run_simulations(N, EXP)

    EXP.agent_constructor = ida.NeighborAgent
    nbr_metrics = run_simulations(N, EXP)

    EXP.agent_constructor = ida.RandomNeighbor
    rnbr_metrics = run_simulations(N, EXP)

    metrics = [null_metrics, mid_metrics, rng_metrics, nbr_metrics, rnbr_metrics]
    labels = ["No treatment", "Preemptive", "Random", "Precision", "Precision: Random"]
    plt.figure()
    plt.ylabel("Average number of sick days")
    plt.xticks(rotation=45)
    for metric, label in zip(metrics, labels):
        total_sickdays = metric["total_sickdays"] / N
        plt.bar(label, total_sickdays)
    plt.tight_layout()
    plt.savefig("out/chain-sickdays.png")

    marks = ["-*", "-v", "-^", "-o", "-s"]
    plt.figure()
    plt.xlabel("Node number")
    plt.ylabel("Probability of infection")
    plt.ylim(bottom=0.0, top=0.7)
    for metric, label, mark in zip(metrics, labels, marks):
        num_sickdays = metric["num_sickdays"] / N
        x = range(num_sickdays.shape[0])
        y = num_sickdays
        plt.plot(x, y, mark, label=label)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("out/chain-probinfect.png")