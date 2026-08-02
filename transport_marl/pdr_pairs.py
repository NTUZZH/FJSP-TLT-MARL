"""PDR rule pairs for FJSP-TL-T baselines (proposal Appendix B):
machine side {SPT, MWKR, FIFO} x vehicle side {STT, NVF, FIFO} = 9 combos.
All rules deterministic with pinned tie-breaks (value, job, machine/vehicle idx).
"""

import numpy as np


def _min_elig_pt(sim):
    return np.array([sim.op_pt[o][sim.op_pt[o] > 0].min()
                     for o in range(sim.n_ops)])


def _remaining_work(sim, job, min_pt):
    o = sim.next_op[job]
    if o < 0:
        return 0.0
    return float(min_pt[o:sim.last_op[job] + 1].sum())


# ---- machine-side rules: (sim, pairs[(j, m, o)]) -> (j, m) ----

def mch_spt(sim, pairs):
    return min(pairs, key=lambda p: (sim.op_pt[p[2], p[1]], p[0], p[1]))[:2]


def mch_mwkr(sim, pairs):
    min_pt = _min_elig_pt(sim)
    return min(pairs, key=lambda p: (-_remaining_work(sim, p[0], min_pt),
                                     p[0], sim.op_pt[p[2], p[1]], p[1]))[:2]


def mch_fifo(sim, pairs):
    return min(pairs, key=lambda p: (sim.job_ready[p[0]], p[0],
                                     sim.op_pt[p[2], p[1]], p[1]))[:2]


# ---- vehicle-side rules: (sim, veh, tasks[dict]) -> job ----

def veh_stt(sim, veh, tasks):
    loc = sim.veh_loc[veh]
    return min(tasks, key=lambda t: (sim.tau[loc, t['frm']] + sim.tau[t['frm'], t['to']],
                                     t['job']))['job']


def veh_nvf(sim, veh, tasks):
    loc = sim.veh_loc[veh]
    return min(tasks, key=lambda t: (sim.tau[loc, t['frm']], t['job']))['job']


def veh_fifo(sim, veh, tasks):
    return min(tasks, key=lambda t: (t['release'], t['job']))['job']


MCH_RULES = {'SPT': mch_spt, 'MWKR': mch_mwkr, 'FIFO': mch_fifo}
VEH_RULES = {'STT': veh_stt, 'NVF': veh_nvf, 'FIFO': veh_fifo}


def env_veh_rule_action(env, e, rule):
    """Vehicle rule applied to a BATCHED env's env e (same keys/tie-breaks as
    the sim rules above; used by E0b training and fixed-rule evaluation)."""
    v = env.event_veh[e]
    loc = env.veh_cell[e, v]
    keys = []
    for j in np.nonzero(env.task_active[e])[0]:
        frm, to = env.task_from[e, j], env.task_to[e, j]
        if rule == 'STT':
            key = (env.tau_cells[e, loc, frm] + env.tau_cells[e, frm, to], j)
        elif rule == 'NVF':
            key = (env.tau_cells[e, loc, frm], j)
        else:  # FIFO
            key = (env.task_release[e, j], j)
        keys.append((key, j))
    return int(min(keys)[1])


def run_pdr_pair(sim, mch_name, veh_name):
    sim.reset()
    return sim.run(MCH_RULES[mch_name], VEH_RULES[veh_name])
