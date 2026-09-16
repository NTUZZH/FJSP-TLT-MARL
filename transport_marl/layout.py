"""Station layout and travel-time model for FJSP-TL-T (Paper X2, design_v1 §1/§6).

v1 pins:
- 9 PPVC station types A..H,Q occupy a 3x3 zone grid in process-flow order;
  machines of the same type are co-located (tau = 0 within a zone).
- tau(a, b) = Manhattan(cell_a, cell_b) * speed_const;  tau_e = tau.
- speed_const is calibrated per instance so that the realized travel intensity
  tau_bar / p_bar hits a target, where
      tau_bar = mean over MANDATORY consecutive-op moves of the minimal
                eligible-pair travel at unit speed  (a move is mandatory iff
                every eligible station pair for the transition has tau > 0),
      p_bar   = mean over ops of the mean eligible processing time.
"""

import numpy as np

# 3x3 zone grid, process-flow order (A=mould head, Q=QC tail).
ZONE_GRID = {
    'A': (0, 0), 'B': (0, 1), 'C': (0, 2),
    'D': (1, 0), 'E': (1, 1), 'F': (1, 2),
    'G': (2, 0), 'H': (2, 1), 'Q': (2, 2),
}
N_CELLS = 9
STATION_TYPES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Q']  # cell i = type i
VEH_START_CELL = 0  # all vehicles start at the 'A' zone (mould line head)

CELL_XY = np.array([ZONE_GRID[t] for t in STATION_TYPES], dtype=float)  # [9,2]
UNIT_TAU = np.abs(CELL_XY[:, None, :] - CELL_XY[None, :, :]).sum(-1)    # [9,9] Manhattan


def station_cells_from_mch_type(mch_type):
    """PPVC: a machine's cell is its station type's zone cell."""
    return np.asarray(mch_type, dtype=int).copy()


def eligible_sets(job_length, op_pt):
    """Per-op eligible machine index arrays (op_pt > 0)."""
    n_ops = int(np.sum(job_length))
    return [np.nonzero(op_pt[i] > 0)[0] for i in range(n_ops)]


def consecutive_pairs(job_length):
    """List of (pred_op, succ_op) global-index pairs within each job."""
    pairs, start = [], 0
    for L in job_length:
        for k in range(int(L) - 1):
            pairs.append((start + k, start + k + 1))
        start += int(L)
    return pairs


def min_unit_travel_per_pair(job_length, op_pt, station_cell):
    """For each consecutive-op pair: min over eligible (m_pred, m_succ) of
    UNIT_TAU[cell(m_pred), cell(m_succ)] (0 if a same-cell assignment exists).
    Returns (pairs, min_unit_tau [len(pairs)], mandatory [len(pairs)] bool)."""
    elig = eligible_sets(job_length, op_pt)
    pairs = consecutive_pairs(job_length)
    out = np.zeros(len(pairs))
    mandatory = np.zeros(len(pairs), dtype=bool)
    for idx, (a, b) in enumerate(pairs):
        ca = station_cell[elig[a]]
        cb = station_cell[elig[b]]
        m = UNIT_TAU[np.ix_(ca, cb)].min()
        out[idx] = m
        mandatory[idx] = m > 0
    return pairs, out, mandatory


def calibrate_speed(job_length, op_pt, station_cell, target_ratio):
    """speed_const s.t. mean mandatory min-travel / mean mean-eligible-pt ~= target.

    The result is snapped to the nearest positive multiple of 1/128 (dyadic
    rational): every travel time is then exactly representable in float64,
    so the reference simulator, the batched env, and the CP-SAT model
    (fix-point SCALE=128) break ties identically. Pinned in decisions.md."""
    _, min_tau_unit, mandatory = min_unit_travel_per_pair(job_length, op_pt, station_cell)
    if not mandatory.any() or target_ratio <= 0:
        return 0.0
    tau_bar_unit = float(min_tau_unit[mandatory].mean())
    pt = np.asarray(op_pt, dtype=float)
    p_bar = float(np.array([pt[i][pt[i] > 0].mean() for i in range(pt.shape[0])]).mean())
    raw = target_ratio * p_bar / tau_bar_unit
    return max(1.0, round(raw * 128.0)) / 128.0


def build_transport_layout(job_length, op_pt, mch_type, target_ratio, n_vehicles):
    """Assemble the transport side-car dict stored in meta.json (design_v1 §6)."""
    station_cell = station_cells_from_mch_type(mch_type)
    speed = calibrate_speed(job_length, op_pt, station_cell, target_ratio)
    tau_cells = UNIT_TAU * speed
    return {
        'station_cell': station_cell.tolist(),
        'cell_xy': CELL_XY.tolist(),
        'speed_const': speed,
        'tau_cells': tau_cells.tolist(),
        'n_vehicles': int(n_vehicles),
        'veh_start_cell': VEH_START_CELL,
        'tau_over_p': float(target_ratio),
    }
