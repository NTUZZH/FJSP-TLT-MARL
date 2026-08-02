"""Lazy-agent / coordination diagnostics for E2 (proposal §8, F3 support).

From a schedule record (sim_single / env format): per-vehicle and per-machine
busy shares, idle rates, contribution Gini, and transport service latency
(release -> pickup wait: how long jobs sat waiting for a vehicle).
"""

import numpy as np


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def agent_diagnostics(record, op_pt, n_vehicles, n_machines):
    ms = float(record['makespan'])
    amch = np.asarray(record['assigned_mch'])
    starts = np.asarray(record['op_start'])
    cts = np.asarray(record['op_ct'])
    veh_busy = np.zeros(n_vehicles)
    waits = []
    for tr in record['transports']:
        if 0 <= tr['veh'] < n_vehicles:
            veh_busy[tr['veh']] += tr['arrival'] - tr['depart']
        # release is not stored in the record; pickup - depart = empty move,
        # so job wait = pickup - release is approximated by depart-side wait
        # only when release is present (env exports it via task history);
        # transports carry depart/pickup/arrival: report pickup latency
        waits.append(tr['pickup'] - tr['depart'])
    mch_busy = np.zeros(n_machines)
    for o in range(len(amch)):
        mch_busy[amch[o]] += cts[o] - starts[o]
    return dict(
        makespan=ms,
        veh_busy=veh_busy.tolist(),
        veh_idle_rate=(1.0 - veh_busy / ms).tolist() if ms > 0 else None,
        veh_gini=gini(veh_busy),
        mch_busy_total=float(mch_busy.sum()),
        mch_idle_rate=float(1.0 - mch_busy.mean() / ms) if ms > 0 else None,
        mch_gini=gini(mch_busy),
        empty_move_mean=float(np.mean(waits)) if waits else 0.0,
        n_transports=len(record['transports']),
    )
