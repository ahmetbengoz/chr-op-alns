# ============================================
# CMAS-CHR-OP — REAL PIPELINE (EJOR-STYLE)
# - Event-based picker–robot synchronization (handover meetings)
# - Time-expanded reservation (node+edge, cap=1) for congestion
# - ALNS on decision variables (handover choices, robot assignment, robot routing)
# - Benchmarks + Excel outputs + Figure 2 + Figure 3 (CI)
# ============================================

import os
import math
import random
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Reproducibility
# -----------------------------
MASTER_SEED = 123
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

# -----------------------------
# Experiment design (adjustable)
# -----------------------------
AISLES = [10, 20, 40]
ITEMS  = [200, 1000]                 # add 5000 later if needed (runtime grows)
DENS   = ["low", "medium", "high"]
SEEDS_PER = 5                        # increase later if needed

P_LIST = [2, 5, 10]
R_LIST = [1, 3, 5, 10]

# -----------------------------
# Key parameters (FINAL — as requested)
# -----------------------------
AISLE_LEN = 20                       # y=1..L-1 storage, y=0 and y=L are cross-aisles/handovers
CAP = 1                              # narrow-aisle capacity
SERVICE = 4                          # meeting service time (time steps)
BATCH_SIZE = 4                       # picker batching
V_PICK = 1.0
V_ROB  = 1.2                          # robot speed multiplier (can be >1)

# -----------------------------
# ALNS budget
# -----------------------------
ALNS_ITERS = 1500
T0 = 40.0
COOL = 0.997

# Internal scalarization parameter used inside ALNS acceptance/objective.
# Keep as-is if you want EXACT behavior consistent with the latest pipeline runs.
LAMBDA_WAIT = 0.02

# -----------------------------
# Output folders (repo-friendly)
# -----------------------------
OUT_DATA = "data"
OUT_FIGS = "figures"
OUT_LOGS = "logs"
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_LOGS, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def gen_items(A, L, n, density, seed):
    # local RNG for deterministic instance generation
    rnd_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)

    if density == "high":
        hot = random.sample(range(1, A+1), k=max(1, A//5))
        xs = np.random.choice(hot, size=n, replace=True)
    elif density == "medium":
        hot = random.sample(range(1, A+1), k=max(1, A//4))
        xs = np.where(
            np.random.rand(n) < 0.55,
            np.random.choice(hot, size=n, replace=True),
            np.random.randint(1, A+1, size=n)
        )
    else:
        xs = np.random.randint(1, A+1, size=n)

    ys = np.random.randint(1, L, size=n)  # 1..L-1
    coords = list(zip(xs.tolist(), ys.tolist()))

    random.setstate(rnd_state)
    np.random.set_state(np_state)
    return coords

def choose_handover_for_point(pt, L):
    x, y = pt
    return (x, 0) if y <= L/2 else (x, L)

# -----------------------------
# Reservation-based movement (node+edge)
# time step = 1
# node occupancy cap=1
# edge occupancy prevents head-on swap: reserve (u,v,t) and (v,u,t)
# -----------------------------
class Reservation:
    def __init__(self, cap=1):
        self.cap = cap
        self.node_occ = defaultdict(int)  # (node,t)->count
        self.edge_occ = set()             # (u,v,t)

    def can_enter(self, v, t):
        return self.node_occ[(v, t)] < self.cap

    def can_traverse(self, u, v, t):
        return ((u, v, t) not in self.edge_occ) and ((v, u, t) not in self.edge_occ)

    def reserve_step(self, u, v, t):
        self.node_occ[(v, t+1)] += 1
        self.edge_occ.add((u, v, t))

    def move_one(self, u, v, t):
        while not (self.can_traverse(u, v, t) and self.can_enter(v, t+1)):
            t += 1
        self.reserve_step(u, v, t)
        return t+1

def manhattan_path(u, v):
    (x1, y1), (x2, y2) = u, v
    path = [u]
    x, y = x1, y1

    dx = 1 if x2 > x else -1
    while x != x2:
        x += dx
        path.append((x, y))

    dy = 1 if y2 > y else -1
    while y != y2:
        y += dy
        path.append((x, y))

    return path

def travel_with_reservation(res: Reservation, u, v, t0, speed=1.0):
    path = manhattan_path(u, v)
    t = t0

    if speed <= 1.000001:
        for i in range(len(path)-1):
            t = res.move_one(path[i], path[i+1], t)
        return t

    # speed>1: allow multiple discrete moves per tick (approximation consistent with discrete reservations)
    k = int(speed)
    frac = speed - k
    i = 0
    while i < len(path)-1:
        moves_this_tick = k
        if random.random() < frac:
            moves_this_tick += 1
        for _ in range(moves_this_tick):
            if i >= len(path)-1:
                break
            t = res.move_one(path[i], path[i+1], t)
            i += 1
    return t

# -----------------------------
# Picker batching and assignment
# -----------------------------
def assign_items_to_pickers(coords, P):
    order = sorted(range(len(coords)), key=lambda i: (coords[i][0], coords[i][1]))
    buckets = [[] for _ in range(P)]
    for k, idx in enumerate(order):
        buckets[k % P].append(idx)
    return buckets

def picker_batches(coords, idxs, batch_size):
    seq = [coords[i] for i in idxs]
    return [seq[i:i+batch_size] for i in range(0, len(seq), batch_size)]

# -----------------------------
# Solution structure
# -----------------------------
def build_initial_solution(inst, P, R):
    coords = inst["coords"]
    L = inst["L"]

    buckets = assign_items_to_pickers(coords, P)
    batches = {}
    batch_h = {}
    load = defaultdict(int)

    for p in range(P):
        b = picker_batches(coords, buckets[p], BATCH_SIZE)
        batches[p] = b
        for bi, items in enumerate(b):
            last = items[-1]
            h = choose_handover_for_point(last, L)
            batch_h[(p, bi)] = h
            load[h] += 1

    # assign handovers to robots by load balancing
    robots_h = {r: [] for r in range(R)}
    rload = {r: 0 for r in range(R)}
    for h, l in sorted(load.items(), key=lambda x: x[1], reverse=True):
        rr = min(rload, key=lambda k: rload[k])
        robots_h[rr].append(h)
        rload[rr] += l

    # initial robot routes: sort by (x,y)
    robot_route = {r: sorted(robots_h[r], key=lambda t: (t[0], t[1])) for r in range(R)}

    return {
        "batches": batches,
        "batch_h": dict(batch_h),
        "robots_h": {r: list(hs) for r, hs in robots_h.items()},
        "robot_route": {r: list(rt) for r, rt in robot_route.items()},
    }

def deepcopy_sol(sol):
    return {
        "batches": sol["batches"],  # static per (inst,P)
        "batch_h": dict(sol["batch_h"]),
        "robots_h": {r: list(hs) for r, hs in sol["robots_h"].items()},
        "robot_route": {r: list(rt) for r, rt in sol["robot_route"].items()},
    }

# -----------------------------
# Event-based synchronized simulation
# -----------------------------
def simulate(inst, P, R, sol, sync=True, congestion=True):
    L = inst["L"]
    coords = inst["coords"]
    depot = (0, 0)

    # ensure batch_h is complete
    for p in range(P):
        b = sol["batches"][p]
        for bi in range(len(b)):
            if (p, bi) not in sol["batch_h"]:
                last = b[bi][-1]
                sol["batch_h"][(p, bi)] = choose_handover_for_point(last, L)

    res = Reservation(cap=CAP if congestion else 999999)

    # meetings queue per handover
    meetings_at = defaultdict(list)

    picker_finish = [0] * P
    robot_finish = [0] * R
    robot_wait_total = 0

    # PICKERS
    for p in range(P):
        t = 0
        cur = depot
        batches = sol["batches"][p]

        for bi, items in enumerate(batches):
            remaining = items.copy()
            while remaining:
                nxt = min(remaining, key=lambda v: abs(v[0]-cur[0]) + abs(v[1]-cur[1]))
                remaining.remove(nxt)
                t = travel_with_reservation(res, cur, nxt, t, speed=V_PICK)
                cur = nxt

            h = sol["batch_h"][(p, bi)]
            t = travel_with_reservation(res, cur, h, t, speed=V_PICK)
            cur = h

            meetings_at[h].append([p, t, bi])

        t = travel_with_reservation(res, cur, depot, t, speed=V_PICK)
        picker_finish[p] = t

    # sort meeting queues by arrival
    for h in meetings_at:
        meetings_at[h].sort(key=lambda x: x[1])

    # ROBOTS
    for r in range(R):
        t = 0
        cur = depot
        route = sol["robot_route"][r]

        for h in route:
            t = travel_with_reservation(res, cur, h, t, speed=V_ROB)
            cur = h

            if not sync:
                continue

            q = meetings_at.get(h, [])
            while q:
                p, parr, bi = q[0]
                if t < parr:
                    robot_wait_total += (parr - t)
                    t = parr
                t += SERVICE
                q.pop(0)

        t = travel_with_reservation(res, cur, depot, t, speed=V_ROB)
        robot_finish[r] = t

    makespan = max(max(picker_finish), max(robot_finish))
    total_wait = robot_wait_total
    return makespan, total_wait, robot_wait_total

# -----------------------------
# Baselines
# -----------------------------
def eval_method(inst, P, R, sol, method):
    if method == "Integrated":
        return simulate(inst, P, R, sol, sync=True,  congestion=True)
    if method == "NoCongestion":
        return simulate(inst, P, R, sol, sync=True,  congestion=False)
    if method == "NoSync":
        return simulate(inst, P, R, sol, sync=False, congestion=True)
    if method == "Sequential":
        return simulate(inst, P, R, sol, sync=False, congestion=False)
    raise ValueError("Unknown method")

# -----------------------------
# ALNS utilities
# -----------------------------
def proxy_batch_cost(last_item, h, depot=(0, 0)):
    return manhattan(last_item, h) + manhattan(h, depot)

def rebuild_robots_h_from_batchh(sol, R):
    load = defaultdict(int)
    for h in sol["batch_h"].values():
        load[h] += 1
    robots_h = {r: [] for r in range(R)}
    rload = {r: 0 for r in range(R)}
    for h, l in sorted(load.items(), key=lambda x: x[1], reverse=True):
        rr = min(rload, key=lambda k: rload[k])
        robots_h[rr].append(h)
        rload[rr] += l
    sol["robots_h"] = {r: list(hs) for r, hs in robots_h.items()}

def route_cost(route, depot=(0, 0)):
    if not route:
        return 0
    cur = depot
    c = 0
    for h in route:
        c += manhattan(cur, h)
        cur = h
    c += manhattan(cur, depot)
    return c

def regret_insert_route(handovers, depot=(0, 0), k=2):
    if not handovers:
        return []
    route = [handovers[0]]
    remaining = handovers[1:]

    while remaining:
        best_choice = None
        best_regret = -1e18

        for h in remaining:
            costs = []
            for pos in range(len(route)+1):
                cand = route[:pos] + [h] + route[pos:]
                costs.append(route_cost(cand, depot))
            costs_sorted = sorted(costs)
            regret = costs_sorted[min(k-1, len(costs_sorted)-1)] - costs_sorted[0]

            if regret > best_regret:
                best_regret = regret
                best_choice = (h, costs.index(min(costs)), min(costs))

        h, pos, _ = best_choice
        route = route[:pos] + [h] + route[pos:]
        remaining.remove(h)

    return route

class ALNSRunner:
    def __init__(self, inst, P, R):
        self.inst = inst
        self.P = P
        self.R = R
        self.cur = build_initial_solution(inst, P, R)
        self.best = deepcopy_sol(self.cur)
        self.best_val = None

        self.destroy_ops = ["RandomBatchH", "HotspotH", "RobotRouteShake"]
        self.repair_ops  = ["GreedyH", "RebalanceRobots", "RegretInsertRoute"]

        self.wd = {op: 1.0 for op in self.destroy_ops}
        self.wr = {op: 1.0 for op in self.repair_ops}

    def pick(self, w):
        ops = list(w.keys())
        ww = np.array([w[o] for o in ops], float)
        ww = ww / ww.sum()
        return np.random.choice(ops, p=ww)

    def destroy(self, sol, op):
        removed_batches = []

        if op == "RandomBatchH":
            keys = list(sol["batch_h"].keys())
            m = max(1, int(0.12 * len(keys)))
            removed_batches = random.sample(keys, m)
            for k in removed_batches:
                sol["batch_h"].pop(k, None)

        elif op == "HotspotH":
            load = defaultdict(int)
            for k, h in sol["batch_h"].items():
                load[h] += 1
            if load:
                hmax = max(load, key=lambda h: load[h])
                keys = [k for k, h in sol["batch_h"].items() if h == hmax]
                m = max(1, int(0.25 * len(keys)))
                removed_batches = random.sample(keys, m)
                for k in removed_batches:
                    sol["batch_h"].pop(k, None)

        elif op == "RobotRouteShake":
            r = random.randrange(self.R)
            rt = sol["robot_route"][r]
            if len(rt) >= 4:
                i = random.randint(0, len(rt)-2)
                j = random.randint(i+1, min(len(rt), i+1+max(2, len(rt)//4)))
                del rt[i:j]
                sol["robot_route"][r] = rt

        return removed_batches

    def repair(self, sol, removed_batches, op):
        L = self.inst["L"]

        if op == "GreedyH":
            for (p, bi) in removed_batches:
                last = sol["batches"][p][bi][-1]
                x, _ = last
                h1 = (x, 0)
                h2 = (x, L)
                c1 = proxy_batch_cost(last, h1)
                c2 = proxy_batch_cost(last, h2)
                sol["batch_h"][(p, bi)] = h1 if c1 <= c2 else h2

        elif op == "RebalanceRobots":
            rebuild_robots_h_from_batchh(sol, self.R)

        elif op == "RegretInsertRoute":
            for r in range(self.R):
                hs = list(set(sol["robots_h"].get(r, [])))
                if not hs:
                    sol["robot_route"][r] = []
                else:
                    random.shuffle(hs)
                    sol["robot_route"][r] = regret_insert_route(hs, depot=(0, 0), k=2)

        rebuild_robots_h_from_batchh(sol, self.R)
        return sol

    def objective(self, makespan, total_wait):
        # scalarization used inside ALNS acceptance; consistent with pipeline outputs
        return makespan + LAMBDA_WAIT * total_wait
        def run(self, iters=ALNS_ITERS):
        mk, tw, _ = simulate(self.inst, self.P, self.R, self.cur, sync=True, congestion=True)
        cur_val = self.objective(mk, tw)

        self.best = deepcopy_sol(self.cur)
        self.best_val = cur_val
        best_mk, best_tw = mk, tw

        T = T0
        hist = []

        for it in range(1, iters+1):
            d = self.pick(self.wd)
            r = self.pick(self.wr)

            cand = deepcopy_sol(self.cur)
            removed = self.destroy(cand, d)
            cand = self.repair(cand, removed, r)

            mk2, tw2, _ = simulate(self.inst, self.P, self.R, cand, sync=True, congestion=True)
            cand_val = self.objective(mk2, tw2)

            accept = (cand_val <= cur_val) or (random.random() < math.exp(-(cand_val-cur_val)/(T+1e-9)))
            if accept:
                self.cur = cand
                cur_val = cand_val

            if cand_val < self.best_val:
                self.best = deepcopy_sol(cand)
                self.best_val = cand_val
                best_mk, best_tw = mk2, tw2
                self.wd[d] += 2.0
                self.wr[r] += 2.0
            else:
                self.wd[d] += 0.03
                self.wr[r] += 0.03

            if it % 250 == 0:
                for op in self.wd:
                    self.wd[op] = max(0.1, self.wd[op]*0.9)
                for op in self.wr:
                    self.wr[op] = max(0.1, self.wr[op]*0.9)

            if it % 10 == 0:
                hist.append((it, self.best_val, best_mk, best_tw))

            T *= COOL

        hist_df = pd.DataFrame(hist, columns=["iter", "best_obj", "best_makespan", "best_total_wait"])
        return self.best, best_mk, best_tw, hist_df

# -----------------------------
# RUN FULL EXPERIMENT
# -----------------------------
def main():
    instances = []
    iid = 0

    for A in AISLES:
        for n in ITEMS:
            for d in DENS:
                for s in range(SEEDS_PER):
                    iid += 1
                    seed = 10000 + iid*17 + s
                    coords = gen_items(A, AISLE_LEN, n, d, seed)
                    instances.append({
                        "instance_id": f"I{iid:04d}",
                        "seed": seed,
                        "A": A,
                        "L": AISLE_LEN,
                        "n_items": n,
                        "density": d,
                        "coords": coords
                    })

    inst_df = pd.DataFrame([{k: v for k, v in inst.items() if k != "coords"} for inst in instances])

    results = []
    conv_store = []

    t_start = time.time()

    # -----------------------------
    # Baselines on all instances/configs
    # -----------------------------
    for inst in instances:
        for P in P_LIST:
            for R in R_LIST:
                sol0 = build_initial_solution(inst, P, R)

                for m in ["Integrated", "NoCongestion", "NoSync", "Sequential"]:
                    mk, tw, rw = eval_method(inst, P, R, sol0, m)
                    results.append({
                        "instance_id": inst["instance_id"],
                        "A": inst["A"],
                        "n_items": inst["n_items"],
                        "density": inst["density"],
                        "cap": CAP,
                        "pickers": P,
                        "robots": R,
                        "method": m,
                        "makespan": mk,
                        "total_wait": tw,
                        "robot_wait": rw,
                        "cpu_s": 0.0
                    })

    # -----------------------------
    # ALNS on "hard subset" (six hard instances)
    # -----------------------------
    hard_subset = [
        inst for inst in instances
        if (inst["A"] == 40 and inst["density"] in ("medium", "high") and inst["n_items"] == max(ITEMS))
    ]
    hard_subset = hard_subset[:min(6, len(hard_subset))]

    for inst in hard_subset:
        for R in R_LIST:
            runner = ALNSRunner(inst, P=5, R=R)
            t0 = time.time()
            best_sol, mk, tw, hist = runner.run(iters=ALNS_ITERS)
            cpu = time.time() - t0

            results.append({
                "instance_id": inst["instance_id"],
                "A": inst["A"],
                "n_items": inst["n_items"],
                "density": inst["density"],
                "cap": CAP,
                "pickers": 5,
                "robots": R,
                "method": "Integrated-ALNS",
                "makespan": mk,
                "total_wait": tw,
                "robot_wait": tw,
                "cpu_s": cpu
            })

            # store one convergence per instance (use a fixed robot count as representative)
            if R == 3:
                h = hist.copy()
                h["instance_id"] = inst["instance_id"]
                h["robots"] = R
                conv_store.append(h)

    elapsed = time.time() - t_start
    res_df = pd.DataFrame(results)

    # -----------------------------
    # SAVE EXCEL OUTPUTS
    # -----------------------------
    inst_df.to_excel(os.path.join(OUT_DATA, "dataset_instances.xlsx"), index=False)
    res_df.to_excel(os.path.join(OUT_DATA, "results_table.xlsx"), index=False)

    # Table 1: baselines only, aggregated by (pickers, robots, method)
    base = res_df[res_df["method"].isin(["Integrated", "NoCongestion", "NoSync", "Sequential"])].copy()

    table1 = base.groupby(["pickers", "robots", "method"], as_index=False).agg(
        mean_makespan=("makespan", "mean"),
        std_makespan=("makespan", "std"),
        mean_total_wait=("total_wait", "mean"),
        std_total_wait=("total_wait", "std"),
        n=("makespan", "count")
    )
    table1.to_excel(os.path.join(OUT_DATA, "Table1_method_comparison.xlsx"), index=False)

    # Table 2: fleet size effect + ALNS subset summary
    table2_g = res_df[res_df["method"] == "Integrated"].groupby(["pickers", "robots"], as_index=False).agg(
        mean_makespan=("makespan", "mean"),
        std_makespan=("makespan", "std"),
        n=("makespan", "count")
    )

    table2_a = res_df[res_df["method"] == "Integrated-ALNS"].groupby(["pickers", "robots"], as_index=False).agg(
        mean_makespan=("makespan", "mean"),
        std_makespan=("makespan", "std"),
        n=("makespan", "count"),
        mean_cpu_s=("cpu_s", "mean")
    )

    with pd.ExcelWriter(os.path.join(OUT_DATA, "Table2_fleet_size.xlsx")) as w:
        table2_g.to_excel(w, sheet_name="Integrated_All", index=False)
        table2_a.to_excel(w, sheet_name="ALNS_Subset", index=False)

    # -----------------------------
    # FIGURE 2: ALNS convergence (averaged across the six hard instances)
    # -----------------------------
    if conv_store:
        conv = pd.concat(conv_store, ignore_index=True)
        avg = conv.groupby("iter", as_index=False)["best_makespan"].mean()

        plt.figure()
        plt.plot(avg["iter"], avg["best_makespan"])
        plt.xlabel("Iteration")
        plt.ylabel("Average best makespan")
        plt.title("Figure 2. ALNS convergence (averaged across the six hard instances)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_FIGS, "figure2_convergence.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # FIGURE 3: fleet size with 95% CI across 90 instances (cap = 1)
    # -----------------------------
    base_int = res_df[res_df["method"] == "Integrated"].copy()

    plt.figure()
    for p in sorted(base_int["pickers"].unique()):
        sub = base_int[base_int["pickers"] == p]

        agg_mean = sub.groupby("robots")["makespan"].mean().sort_index()
        agg_std  = sub.groupby("robots")["makespan"].std()
        n        = sub.groupby("robots")["makespan"].count()

        ci = 1.96 * (agg_std / np.sqrt(n))
        ci = ci.reindex(agg_mean.index)

        plt.errorbar(
            agg_mean.index,
            agg_mean.values,
            yerr=ci.values,
            capsize=4,
            marker="o",
            linestyle="-",
            label=f"P = {p}"
        )

    plt.xlabel("Number of Robots")
    plt.ylabel("Makespan")
    plt.title("Figure 3. Effect of robot fleet size on makespan (mean ± 95% confidence interval across 90 instances, cap = 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "figure3_fleet_size.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # RUN LOG
    # -----------------------------
    log_path = os.path.join(OUT_LOGS, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Elapsed seconds: {elapsed:.2f}\n")
        f.write(f"Instances: {len(instances)}\n")
        f.write(f"Hard subset (ALNS): {len(hard_subset)}\n")
        f.write(f"ALNS iters: {ALNS_ITERS}\n")
        f.write(f"AISLE_LEN={AISLE_LEN}, CAP={CAP}, SERVICE={SERVICE}, BATCH_SIZE={BATCH_SIZE}, V_PICK={V_PICK}, V_ROB={V_ROB}\n")
        f.write(f"LAMBDA_WAIT (ALNS scalarization) = {LAMBDA_WAIT}\n")

    print("DONE.")
    print("Outputs:")
    print(f"- {os.path.join(OUT_DATA,'dataset_instances.xlsx')}")
    print(f"- {os.path.join(OUT_DATA,'results_table.xlsx')}")
    print(f"- {os.path.join(OUT_DATA,'Table1_method_comparison.xlsx')}")
    print(f"- {os.path.join(OUT_DATA,'Table2_fleet_size.xlsx')}")
    print(f"- {os.path.join(OUT_FIGS,'figure2_convergence.png')}")
    print(f"- {os.path.join(OUT_FIGS,'figure3_fleet_size.png')}")
    print(f"- {log_path}")

if __name__ == "__main__":
    main()
