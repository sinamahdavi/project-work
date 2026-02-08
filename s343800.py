from __future__ import annotations
from typing import List, Tuple, Dict
import networkx as nx
import math

Action = Tuple[int, float]


def solution(p) -> List[Action]:
    G: nx.Graph = p.graph
    n = G.number_of_nodes()

    gold: Dict[int, float] = {i: float(G.nodes[i].get("gold", 0.0)) for i in range(n)}
    cities = [i for i in range(1, n) if gold[i] > 1e-12]
    if not cities:
        return [(0, 0.0)]

    alpha = float(p.alpha)
    beta = float(p.beta)

    # Precompute shortest paths for routing (N<=100 OK)
    path_all = dict(nx.all_pairs_dijkstra_path(G, weight="dist"))
    dist_all = dict(nx.all_pairs_dijkstra_path_length(G, weight="dist"))

    if beta > 1.0:
        acts = _chunk_split_beta_gt_1(G, gold, cities, path_all, dist_all, alpha, beta)
        return _ensure_end_base(acts)

    # beta <= 1: start from baseline, then merge routes only if it strictly improves exact cost
    baseline_routes = [[v] for v in cities]  # each route is a list of cities, base is implicit
    routes = _greedy_merge_routes_beta_le_1(p, G, gold, baseline_routes, path_all)

    acts = _routes_to_actions(routes, gold, path_all)
    return _ensure_end_base(acts)


# beta > 1 : chunk split (return to base between chunks)
def _chunk_split_beta_gt_1(
    G: nx.Graph,
    gold: Dict[int, float],
    cities: List[int],
    path_all,
    dist_all,
    alpha: float,
    beta: float,
) -> List[Action]:
    cities_sorted = sorted(cities, key=lambda v: dist_all[0].get(v, float("inf")), reverse=True)
    actions: List[Action] = []
    MAX_ACTIONS = 250000

    for v in cities_sorted:
        rem = float(gold[v])
        if rem <= 1e-12:
            continue

        d = float(dist_all[0][v])
        path = path_all[0][v]
        back = path[::-1]

        cap = _chunk_size(alpha, beta, d)

        while rem > 1e-9:
            take = min(cap, rem)

            # 0 -> v (pickup at v)
            for node in path[1:-1]:
                actions.append((node, 0.0))
            actions.append((v, take))

            # v -> 0 (deposit)
            for node in back[1:]:
                actions.append((node, 0.0))

            rem -= take
            if len(actions) > MAX_ACTIONS:
                # finish remainder in one last trip to avoid explosion
                if rem > 1e-9:
                    for node in path[1:-1]:
                        actions.append((node, 0.0))
                    actions.append((v, rem))
                    for node in back[1:]:
                        actions.append((node, 0.0))
                break

    return actions


def _chunk_size(alpha: float, beta: float, d: float) -> float:
    alpha = max(alpha, 1e-9)
    d = max(d, 1e-9)
    # beta > 1
    cap = (1.0 / (alpha * d)) * ((2.0 * d) / (beta - 1.0)) ** (1.0 / beta)
    return max(10.0, min(250.0, cap))


# beta <= 1 : greedy merge of routes with exact-cost test
def _greedy_merge_routes_beta_le_1(
    p,
    G: nx.Graph,
    gold: Dict[int, float],
    routes: List[List[int]],
    path_all,
) -> List[List[int]]:
    """
    Start with baseline routes: [[v1],[v2],...]
    Try to merge two routes if merged exact cost is smaller than separate.
    This guarantees NEVER worse than baseline.
    """
    # Precompute exact costs of current routes
    route_costs = [ _route_cost(p, G, gold, r, path_all) for r in routes ]

    MAX_MERGES = 200  # safety
    merges = 0
    improved = True

    while improved and merges < MAX_MERGES and len(routes) >= 2:
        improved = False
        best_gain = 0.0
        best_i = best_j = -1
        best_new_route = None
        best_new_cost = None

        m = len(routes)
        for i in range(m):
            for j in range(i + 1, m):
                r1 = routes[i]
                r2 = routes[j]

                # Try both concatenation orders
                cand1 = r1 + r2
                cand2 = r2 + r1

                c1 = _route_cost(p, G, gold, cand1, path_all)
                c2 = _route_cost(p, G, gold, cand2, path_all)

                old = route_costs[i] + route_costs[j]

                if c1 < old - 1e-9:
                    gain = old - c1
                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i, j
                        best_new_route = cand1
                        best_new_cost = c1

                if c2 < old - 1e-9:
                    gain = old - c2
                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i, j
                        best_new_route = cand2
                        best_new_cost = c2

        if best_new_route is not None:
            # Apply best merge
            i, j = best_i, best_j
            # remove higher index first
            for idx in sorted([i, j], reverse=True):
                routes.pop(idx)
                route_costs.pop(idx)

            routes.append(best_new_route)
            route_costs.append(float(best_new_cost))

            merges += 1
            improved = True

    return routes


def _route_cost(p, G: nx.Graph, gold: Dict[int, float], route: List[int], path_all) -> float:
    """
    Exact cost of a route:
      0 -> route[0] -> route[1] -> ... -> route[k-1] -> 0
    Pick ALL gold at each city when arrived.
    No deposit until final return to 0.
    """
    cur = 0
    carried = 0.0
    total = 0.0

    for v in route:
        path = path_all[cur][v]
        # traverse edges with current carried
        for a, b in zip(path, path[1:]):
            total += float(p.cost([a, b], carried))
        carried += float(gold[v])
        cur = v

    # return to base
    if cur != 0:
        path = path_all[cur][0]
        for a, b in zip(path, path[1:]):
            total += float(p.cost([a, b], carried))

    return float(total)


def _routes_to_actions(routes: List[List[int]], gold: Dict[int, float], path_all) -> List[Action]:
    """
    Convert route list into ACTIONS list.
    Each route starts at 0 and ends at 0 (deposit at end).
    """
    actions: List[Action] = []
    for route in routes:
        cur = 0
        for v in route:
            path = path_all[cur][v]
            for node in path[1:-1]:
                actions.append((node, 0.0))
            actions.append((v, float(gold[v])))
            cur = v

        # return to base
        if cur != 0:
            path = path_all[cur][0]
            for node in path[1:]:
                actions.append((node, 0.0))

    return actions


def _ensure_end_base(actions: List[Action]) -> List[Action]:
    if not actions:
        return [(0, 0.0)]
    if actions[-1][0] != 0:
        actions.append((0, 0.0))
    else:
        actions[-1] = (0, 0.0)
    return actions
