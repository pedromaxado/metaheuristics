from multiprocessing import pool
import numpy as np
import multiprocessing

from os import listdir
from enum import Enum
from typing import Dict, List, Callable, Tuple

import tsplib95 as tsp

instances_dir = "tsp_instances/EUC_2D/"


class GreedyMode(Enum):
    STANDARD = 1,
    GRASP = 2,


def f(x: List[int]) -> int:
    fx = 0

    for i in range(1, len(x)):
        fx += w[x[i-1]][x[i]]

    fx += w[len(x)-1][0]

    return fx


def build_adj_mtx(problem, g_size) -> Dict[int, List[int]]:
    w = dict()

    for u in range(1, g_size + 1):
        w[u] = [0] * (g_size + 1)

        for v in range(1, g_size + 1):
            if u != v:
                w[u][v] = problem.get_weight(u, v)
            else:
                w[u][v] = 10000000

    return w


def hc(mode: GreedyMode, v: int, visited: List[bool], cycle: List[int]) -> int:

    cycle.append(v)

    visited[v] = True

    next_v = 0
    end = True
    best_weight = 10000000

    if mode == GreedyMode.STANDARD:
        for u in range(1, g_size + 1):
            if not visited[u] and w[v][u] < best_weight:
                next_v = u
                end = False
                best_weight = w[v][u]
    else:
        not_visited = [idx for idx in range(1, len(visited)) if not visited[idx]]

        if len(not_visited) > 0:
            if len(not_visited) == 1:
                rand_idx = 0
            else:
                rand_idx = np.random.randint(1, len(not_visited))

            next_v = not_visited[rand_idx]
            end = False
            best_weight = w[v][next_v]

    if end:
        cycle.append(1)
        return w[v][1]

    return best_weight + hc(mode, next_v, visited, cycle)


def search_best_cycle(g_size: int, mode: GreedyMode) -> Tuple[List[int], int]:
    best_fx = 10000000
    best_x = []

    for i in range(1, g_size+1):
        cycle = list()
        visited = [False] * (g_size + 1)
        sol_val = hc(mode, i, visited, cycle)

        if sol_val < best_fx:
            best_fx = sol_val
            best_x = cycle

    return best_x, best_fx


def trivial_solution(g_size: int) -> Tuple[List[int], int]:
    x = [1]
    fx = 0

    for v in range(2, g_size + 1):
        x.append(v)
        fx += w[x[v - 2]][x[v - 1]]

    x.append(1)
    fx += w[g_size][1]

    return x, fx


def is_tour_valid(tour: List[int]) -> bool:

    nodes = []

    for idx in range(len(tour)):
        v = tour[idx]

        if v not in nodes:
            nodes.append(v)
        elif idx < len(nodes) - 1 and v in nodes:
            return False

    return True


def tsp2opt(g_size: int, x0: List[int], fx0: int) -> Tuple[List[int], int]:
    best_x = x0
    best_fx = fx0

    for i in range(1, g_size - 2):
        for j in range(i + 1, g_size - 1):
            if j + 1 >= len(x0):
                continue

            a, b = x0[i - 1], x0[i]
            c, d = x0[j], x0[j + 1]

            old_value = w[a][b] + w[c][d]
            new_value = w[a][c] + w[b][d]

            tmp_x = list(x0)
            tmp_x[i] = c
            tmp_x[j] = b

            tmp_fx = f(tmp_x)

            if is_tour_valid(tmp_x) and tmp_fx < best_fx:
                best_x = tmp_x
                best_fx = tmp_fx

    return best_x, best_fx


def tsp3opt(g_size: int, x0: List[int], fx0: int) -> Tuple[List[int], int]:

    best_x = x0
    best_fx = fx0

    for i in range(1, g_size):
        for j in range(i + 1, g_size):
            for k in range(j + 2, g_size):
                if k+1 >= g_size:
                    continue

                a, b, c, d, e, f = (
                    x0[i - 1],
                    x0[i],
                    x0[j],
                    x0[j+1],
                    x0[k],
                    x0[k+1],
                )

                old_value = w[a][b] + w[c][d] + w[e][f]

                v1 = w[a][c] + w[b][d] + w[e][f]
                v2 = w[a][b] + w[c][e] + w[d][f]
                v3 = w[a][d] + w[e][b] + w[c][f]
                v4 = w[f][b] + w[c][d] + w[e][a]

                if v1 < old_value:
                    tmp_x = list(x0)
                    tmp_x[i] = c
                    tmp_x[j] = b

                    if is_tour_valid(tmp_x):
                        best_x = tmp_x
                        best_fx = fx0 - old_value + v1
                        continue

                if v2 < old_value:
                    tmp_x = list(x0)
                    tmp_x[j+1] = e
                    tmp_x[k] = d

                    if is_tour_valid(tmp_x):
                        best_x = tmp_x
                        best_fx = fx0 - old_value + v2
                        continue

                if v3 < old_value:
                    tmp_x = list(x0)
                    tmp_x[i] = d
                    tmp_x[j] = e
                    tmp_x[j+1] = b
                    tmp_x[k] = c

                    if is_tour_valid(tmp_x):
                        best_x = tmp_x
                        best_fx = fx0 - old_value + v3
                        continue

                if v4 < old_value:
                    tmp_x = list(x0)
                    tmp_x[i-1] = f
                    tmp_x[k+1] = a

                    if is_tour_valid(tmp_x):
                        best_x = tmp_x
                        best_fx = fx0 - old_value + v4
                        continue

    return best_x, best_fx


def vnd(
    g_size: int,
    x0: List[int],
    fx0: int,
    neighborhoods: List[Callable[[int, List[int], int], List[int]]],
) -> Tuple[List[int], int]:

    l = 0
    l_max = len(neighborhoods)

    while l < l_max:
        # print(fx0)
        n = neighborhoods[l]

        x, fx = n(g_size, x0, fx0)

        if fx < fx0:
            x0 = x
            fx0 = fx
        else:
            l += 1

        # print(fx0)

    return x0, fx0


def mh(g_size: int) -> Tuple[List[int], int]:
    neighborhoods = [tsp2opt]

    x0, fx0 = search_best_cycle(g_size)

    best_x, best_fx = vnd(g_size, x0, fx0, neighborhoods)

    return best_x, best_fx


def run_grasp(run: int) -> Tuple[List[int], int]:
    x0, fx0 = search_best_cycle(g_size, GreedyMode.GRASP)
    x, fx = vnd(g_size, x0, fx0, neighborhoods=[tsp2opt])

    return x, fx


def grasp(g_size: int) -> Tuple[List[int], int]:
    x0, fx0 = search_best_cycle(g_size, GreedyMode.STANDARD)

    best_x, best_fx = vnd(g_size, x0, fx0, neighborhoods=[tsp2opt])

    pool_obj = multiprocessing.Pool()

    answers = pool_obj.map(run_grasp, range(0, 10))

    for ans in answers:
        if ans[1] < best_fx:
            best_x = ans[0]
            best_fx = ans[1]
        
    return best_x, best_fx


if __name__ == "__main__":
    global w

    instances = sorted(listdir(instances_dir))
    # instances = ["berlin52.tsp"]

    print(instances)

    for instance in instances:

        problem = tsp.load(instances_dir + instance)
        g_size = len(list(problem.get_nodes()))
        w = build_adj_mtx(problem, g_size)

        print(instance + ": ", end="")

        best_x, best_fx = grasp(g_size)

        print(best_fx)
