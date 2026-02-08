import sys
import os
import random
import statistics
import time
from typing import List, Tuple

import networkx as nx

print("COMPARE STARTED")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

Action = Tuple[int, float]


def simulate_actions_cost(P, actions: List[Action]) -> float:
    """Evaluate a solution in ACTION format: [(next_node, pickup_amount), ...]"""
    G = P.graph
    n = G.number_of_nodes()
    remaining = {i: float(G.nodes[i].get("gold", 0.0)) for i in range(n)}

    cur = 0
    carried = 0.0
    total = 0.0

    for nxt, take in actions:
        nxt = int(nxt)
        take = float(take)

        if nxt != cur:
            if not G.has_edge(cur, nxt):
                return float("inf")
            total += float(P.cost([cur, nxt], carried))

        if take < -1e-9:
            return float("inf")
        if take > remaining.get(nxt, 0.0) + 1e-6:
            return float("inf")

        remaining[nxt] -= take
        carried += take
        cur = nxt

        if cur == 0:
            carried = 0.0

    if cur != 0:
        return float("inf")

    if sum(max(0.0, remaining[i]) for i in range(1, n)) > 1e-5:
        return float("inf")

    return float(total)


def baseline_actions(P) -> List[Action]:
    """Feasible baseline: visit each city, pick all gold, return to base, using shortest paths."""
    G = P.graph
    n = G.number_of_nodes()
    _, paths = nx.single_source_dijkstra(G, source=0, weight="dist")

    acts: List[Action] = []
    for v in range(1, n):
        path = paths[v]
        back = path[::-1]

        for node in path[1:-1]:
            acts.append((node, 0.0))
        acts.append((v, float(G.nodes[v].get("gold", 0.0))))

        for node in back[1:]:
            acts.append((node, 0.0))

    if not acts:
        return [(0, 0.0)]
    acts[-1] = (0, 0.0)
    return acts


def evaluate(seed: int, n: int, density: float, alpha: float, beta: float):
    # Import INSIDE so failures show clearly and do not kill the whole file silently
    from Problem import Problem
    from s343800 import solution

    P = Problem(n, density=density, alpha=alpha, beta=beta, seed=seed)

    base_act = baseline_actions(P)
    base_cost = simulate_actions_cost(P, base_act)

    t0 = time.time()
    sol_act = solution(P)
    t1 = time.time()

    sol_cost = simulate_actions_cost(P, sol_act)

    if base_cost == 0 or base_cost == float("inf"):
        imp = float("nan")
    else:
        imp = 100.0 * (base_cost - sol_cost) / base_cost

    return base_cost, sol_cost, imp, (t1 - t0), len(sol_act)


def main():
    print("MAIN RUNNING\n")

    random.seed(0)
    imps = []

    # Start with 10; change to 50 when stable
    NUM_TESTS = 50

    for seed in range(NUM_TESTS):
        print(f"\n--- Test {seed:02d} ---")

        n = random.choice([20, 30, 50, 80, 100])
        density = random.uniform(0.2, 1.0)
        alpha = random.uniform(0.5, 5.0)
        beta = random.uniform(0.5, 5.0)

        print(f"Params: N={n}, dens={density:.2f}, a={alpha:.2f}, b={beta:.2f}")
        print("Calling evaluate()...")

        base_cost, sol_cost, imp, secs, act_len = evaluate(seed, n, density, alpha, beta)
        imps.append(imp)

        print(
            f"{seed:02d}  N={n:3d}  dens={density:.2f}  a={alpha:.2f}  b={beta:.2f}  "
            f"baseline={base_cost:.3e}  sol={sol_cost:.3e}  imp={imp:+.2f}%  "
            f"time={secs:.2f}s  actions={act_len}"
        )

    finite = [x for x in imps if x == x and abs(x) != float("inf")]

    print("\nSummary")
    if finite:
        print(f"mean improvement: {statistics.mean(finite):+.2f}%")
        print(f"median improvement: {statistics.median(finite):+.2f}%")
        print(f"min/max improvement: {min(finite):+.2f}% / {max(finite):+.2f}%")
    else:
        print("No finite results (all solutions infeasible).")


if __name__ == "__main__":
    main()
