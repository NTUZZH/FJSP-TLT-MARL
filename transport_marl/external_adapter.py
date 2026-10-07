"""Adapters for PUBLIC FJSP-with-transport benchmarks -> our transport stack.

Loads external, machine-readable FJSP-T instances (github.com/lucasberter/FJSPT,
cloned READ-ONLY under ext_benchmarks/FJSPT/) into the exact tuple our reference
simulator (transport_marl/sim_single.TransportSim) and batched env
(transport_marl/fjsp_env_transport.FJSPEnvTransport.set_initial_data) consume:

    job_length : list[int]          # ops per job
    op_pt      : ndarray [N, M]     # processing time, 0 == incompatible machine
    time_lag   : ndarray [N]        # inter-op lags (external sets have none -> 0)
    layout     : dict               # station_cell[M], tau_cells[C,C] (TRUE units),
                                    #   n_vehicles, veh_start_cell, cell_xy[C,2]

Purpose: an EXTERNAL comparability track for Paper X2 E1, so our PDR/RL numbers
sit next to published FJSP-T results on the same instances.

Two formats are supported (they differ in the operation encoding -- do not
confuse them):

1. Deroussi-Norre  (ext_benchmarks/FJSPT/DeroussiNorre/fjsp{1..10}.txt)
   Line 1 : "n_jobs  n_machines  n_vehicles"
   Job line: "n_ops  [ n_alt  m1 m2 ... m_{n_alt}  pt ]  ..."   (1-based machine ids)
   All n_alt alternative machines of an operation share the SAME pt: these are
   DUPLICATED machines. The set is built (Deroussi & Norre 2010) from the
   Bilge-Ulusoy (1995) 4-machine job shop by duplicating each machine, so every
   op has 2 alternatives {2w-1, 2w} = the two copies of Bilge-Ulusoy workstation
   w.  Bilge-Ulusoy has only 4 machine LOCATIONS, so the two copies are
   necessarily CO-LOCATED; hence station_cell maps machine m -> workstation
   cell (m // 2) + 1, with the L/U depot at cell 0.  See CAVEATS below.

2. Berterottiere / Dauzere  (Dauzere_Data/Text/{01..18}a.txt  +
   BerterottiereTravelTimes/layout{5,8,10}.txt  + Berterottiere/dpp{2,4,6}veh/*)
   Standard Brandimarte/Dauzere FJS format:
   Line 1 : "n_jobs  n_machines  avg_alt"
   Job line: "n_ops  [ n_alt  (m1 pt1) (m2 pt2) ... ]  ..."     (1-based ids)
   Here every machine is a distinct physical resource with its OWN location, so
   station_cell is the identity (machine m -> cell m+1) and the shipped travel
   layout (an (M+1)x(M+1) matrix over M machine locations + 1 L/U depot) is used
   verbatim as tau_cells.  n_vehicles comes from the chosen dpp{k}veh folder.

===========================  SEMANTIC CAVEATS  ===============================
These are NOT silently bridged:

C1 (Deroussi-Norre travel matrix is NOT shipped). The lucasberter repo ships
   the Deroussi-Norre PROCESSING data only. The accompanying travel matrices are
   the Bilge-Ulusoy (1995) layouts, published solely as tables in Oper. Res.
   43(6):1058-1070 -- no verified machine-readable copy is in the repo. We
   therefore attach a DETERMINISTIC, CLEARLY-LABELED SYNTHETIC layout (a unit
   grid over depot + 4 workstations) unless a real matrix is passed in. Every
   instance loaded this way carries synthetic_tau=True and MUST NOT be used for
   external makespan comparison -- only for exercising the parse/sim/validate
   pipeline. Pass tau_matrix=<verified Bilge-Ulusoy layout> to make it real.

C2 (No L/U load/unload legs). Berterottiere's objective transports each job
   L/U -> first machine and last machine -> L/U, and may add fixed load/unload
   handling times. Our env only creates a transport on a consecutive-op cell
   CHANGE (the first op materializes at its machine; there is no final return
   to L/U and no handling-time constant). Our makespan is therefore computed on
   a strictly SMALLER objective than the published Cmax: expect our numbers to
   run a bit low relative to a schedule that pays those legs. This is a lower
   biased objective, not a bug.

C3 (Depot index assumption, Berterottiere). The layout is (M+1)x(M+1); we take
   location 0 as the L/U depot (Bilge-Ulusoy heritage) and machines at cells
   1..M. If the source indexes the depot differently, only each vehicle's
   FIRST empty repositioning move differs; inter-machine travel (which drives
   Cmax) is unaffected. Override with depot_index=.

C4 (Deroussi-Norre station mapping vs the "identity" request). For Berterottiere
   station_cell IS the identity (per-machine locations). For Deroussi-Norre a
   literal identity over 8 machine cells would place the two co-located copies
   of a workstation at DIFFERENT cells with nonzero travel between them, which
   contradicts the Bilge-Ulusoy geometry (4 locations). We therefore map
   duplicated machines to their shared workstation cell. This is the ground-truth
   co-location, NOT a PPVC type-zone collapse.
=============================================================================
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np

# Repo layout: this file lives at <root>/transport_marl/external_adapter.py.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_ROOT = os.path.join(_ROOT, 'ext_benchmarks', 'FJSPT')
DN_DIR = os.path.join(EXT_ROOT, 'DeroussiNorre')
DAUZERE_DIR = os.path.join(EXT_ROOT, 'Dauzere_Data', 'Text')
BERT_LAYOUT_DIR = os.path.join(EXT_ROOT, 'BerterottiereTravelTimes')
BERT_RESULT_DIR = os.path.join(EXT_ROOT, 'Berterottiere')


# ---------------------------------------------------------------------------
# Instance container
# ---------------------------------------------------------------------------

@dataclass
class ExternalInstance:
    """One loaded FJSP-T instance in our stack's native representation."""
    name: str
    source: str                       # 'deroussi_norre' | 'berterottiere'
    job_length: list                  # list[int], length n_jobs
    op_pt: np.ndarray                 # [N, M] float, 0 == incompatible
    time_lag: np.ndarray              # [N] float (zeros for external sets)
    station_cell: np.ndarray          # [M] int, cell of each machine
    tau_cells: np.ndarray             # [C, C] float, TRUE units
    n_vehicles: int
    veh_start_cell: int
    cell_xy: np.ndarray               # [C, 2] float (feature-only; not used by sim)
    synthetic_tau: bool = False       # True => tau is a placeholder, NOT publishable
    notes: dict = field(default_factory=dict)

    @property
    def n_jobs(self):
        return len(self.job_length)

    @property
    def n_machines(self):
        return self.op_pt.shape[1]

    @property
    def n_ops(self):
        return int(np.sum(self.job_length))

    def to_sim_args(self):
        """kwargs for transport_marl.sim_single.TransportSim(...)."""
        return dict(job_length=list(self.job_length), op_pt=self.op_pt,
                    time_lag=self.time_lag, station_cell=self.station_cell,
                    tau_cells=self.tau_cells, n_vehicles=int(self.n_vehicles),
                    veh_start_cell=int(self.veh_start_cell))

    def to_validator_args(self, record):
        """positional args for validator_t.validate_transport_schedule(...)."""
        return (list(self.job_length), self.op_pt, self.time_lag,
                self.station_cell, self.tau_cells, int(self.n_vehicles),
                int(self.veh_start_cell), record)

    def layout(self):
        """The transport side-car dict (matches layout.build_transport_layout)."""
        return dict(station_cell=self.station_cell.tolist(),
                    cell_xy=self.cell_xy.tolist(),
                    tau_cells=self.tau_cells.tolist(),
                    n_vehicles=int(self.n_vehicles),
                    veh_start_cell=int(self.veh_start_cell))

    def to_env_lists(self):
        """Length-1 batch lists for FJSPEnvTransport.set_initial_data(...).

        Note: the batched env additionally requires all jobs to share the same
        op count and pt_lower_bound==0; the reference sim has no such constraint,
        so the smoke test drives the sim. These lists let a same-op-nums external
        subset be loaded into the env when needed."""
        return dict(job_length_list=[np.asarray(self.job_length, dtype=int)],
                    op_pt_list=[self.op_pt.astype(float)],
                    time_lag_list=[self.time_lag.astype(float)],
                    op_type_list=None, mch_type_list=None,
                    layout_list=[self.layout()])


# ---------------------------------------------------------------------------
# Low-level parsers
# ---------------------------------------------------------------------------

def _read_int_lines(path):
    """Return non-empty lines as lists of ints (tolerant of extra whitespace)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([int(x) for x in line.split()])
    return rows


def parse_deroussi_norre(path):
    """Parse a Deroussi-Norre fjsp*.txt.

    Returns dict(n_jobs, n_machines, n_vehicles, job_length, op_pt[N,M],
                 op_workstation[N] list-of-eligible-machine-arrays for reference).
    Operation encoding: 'n_alt m1 ... m_{n_alt} pt' -- shared pt across the alts.
    """
    rows = _read_int_lines(path)
    nj, nm, nv = rows[0][0], rows[0][1], rows[0][2]
    job_lines = rows[1:]
    if len(job_lines) != nj:
        raise ValueError(f'{path}: header says {nj} jobs but found {len(job_lines)}')
    job_length, op_pt_rows = [], []
    for jl in job_lines:
        i = 0
        n_ops = jl[i]; i += 1
        job_length.append(n_ops)
        for _ in range(n_ops):
            n_alt = jl[i]; i += 1
            machines = jl[i:i + n_alt]; i += n_alt
            pt = jl[i]; i += 1
            row = np.zeros(nm, dtype=float)
            for m in machines:
                if not (1 <= m <= nm):
                    raise ValueError(f'{path}: machine id {m} out of range 1..{nm}')
                row[m - 1] = pt
            op_pt_rows.append(row)
        if i != len(jl):
            raise ValueError(f'{path}: job line had {len(jl) - i} trailing tokens')
    op_pt = np.stack(op_pt_rows, axis=0)
    return dict(n_jobs=nj, n_machines=nm, n_vehicles=nv,
                job_length=job_length, op_pt=op_pt)


def parse_fjs_standard(path):
    """Parse a standard Brandimarte/Dauzere .fjs/.txt FJSP instance.

    Line 1: 'n_jobs n_machines avg_alt'. Job line: 'n_ops [ n_alt (m pt)... ]...'
    (per-machine pt). Returns dict(n_jobs, n_machines, job_length, op_pt[N,M]).
    """
    rows = _read_int_lines(path)
    nj, nm = rows[0][0], rows[0][1]
    job_lines = rows[1:]
    if len(job_lines) != nj:
        raise ValueError(f'{path}: header says {nj} jobs but found {len(job_lines)}')
    job_length, op_pt_rows = [], []
    for jl in job_lines:
        i = 0
        n_ops = jl[i]; i += 1
        job_length.append(n_ops)
        for _ in range(n_ops):
            n_alt = jl[i]; i += 1
            row = np.zeros(nm, dtype=float)
            for _k in range(n_alt):
                m = jl[i]; pt = jl[i + 1]; i += 2
                if not (1 <= m <= nm):
                    raise ValueError(f'{path}: machine id {m} out of range 1..{nm}')
                row[m - 1] = pt
            op_pt_rows.append(row)
        if i != len(jl):
            raise ValueError(f'{path}: job line had {len(jl) - i} trailing tokens')
    op_pt = np.stack(op_pt_rows, axis=0)
    return dict(n_jobs=nj, n_machines=nm, job_length=job_length, op_pt=op_pt)


def parse_travel_matrix(path):
    """Read a whitespace-separated square travel-time matrix (floats)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split()])
    mat = np.array(rows, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f'{path}: travel matrix is not square: {mat.shape}')
    return mat


def parse_berterottiere_cmax(result_path):
    """Extract the best-reported Cmax from a Berterottiere result file header.

    Header line 1: 'dpp01a_2veh #vehicles: 2 Cmax: 2752.0 #iterations: ...'.
    Returns the smallest Cmax found across the (multiple) reported lines."""
    cmaxes = []
    with open(result_path) as f:
        for line in f:
            m = re.search(r'Cmax:\s*([0-9]+(?:\.[0-9]+)?)', line)
            if m:
                cmaxes.append(float(m.group(1)))
    if not cmaxes:
        raise ValueError(f'{result_path}: no Cmax found')
    return min(cmaxes)


# ---------------------------------------------------------------------------
# Synthetic placeholder layout (Deroussi-Norre only; see caveat C1)
# ---------------------------------------------------------------------------

def _synthetic_workstation_layout(n_ws, scale=6.0):
    """Deterministic PLACEHOLDER travel layout over depot + n_ws workstations.

    NOT the Bilge-Ulusoy matrix. Depot at (0,0); workstations on a compact grid;
    tau = Manhattan(cell_a, cell_b) * scale (integer-friendly). Returns
    (tau_cells[C,C], cell_xy[C,2]) with C = n_ws + 1 and depot = cell 0."""
    xy = [(0.0, 0.0)]                       # cell 0 = depot / L-U
    for w in range(n_ws):                   # workstations on a line offset in x
        xy.append((float(1 + w % 2), float(1 + w // 2)))
    xy = np.array(xy, dtype=float)
    tau = np.abs(xy[:, None, :] - xy[None, :, :]).sum(-1) * float(scale)
    np.fill_diagonal(tau, 0.0)
    return tau, xy


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------

def load_deroussi_norre(path, tau_matrix=None, tau_scale=6.0):
    """Load a Deroussi-Norre instance into an ExternalInstance.

    tau_matrix : optional (n_ws+1)x(n_ws+1) verified Bilge-Ulusoy layout
                 (cell 0 = L/U depot, cells 1..n_ws = workstations). If None,
                 a synthetic placeholder is used and synthetic_tau=True (caveat
                 C1): such an instance is for pipeline validation only.
    """
    p = parse_deroussi_norre(path)
    nm = p['n_machines']
    # Derive workstation grouping empirically: alternatives are duplicated pairs.
    # Machine m (0-based) -> workstation m // 2 (Bilge-Ulusoy: 4 duplicated locs).
    n_ws = (nm + 1) // 2
    ws_of_machine = np.array([m // 2 for m in range(nm)], dtype=int)
    station_cell = ws_of_machine + 1        # depot occupies cell 0

    if tau_matrix is not None:
        tau = np.asarray(tau_matrix, dtype=float)
        C = n_ws + 1
        if tau.shape != (C, C):
            raise ValueError(f'tau_matrix must be {C}x{C} (depot + {n_ws} '
                             f'workstations), got {tau.shape}')
        cell_xy = np.array([[i, 0] for i in range(C)], dtype=float)
        synthetic = False
    else:
        tau, cell_xy = _synthetic_workstation_layout(n_ws, scale=tau_scale)
        synthetic = True

    N = int(np.sum(p['job_length']))
    return ExternalInstance(
        name=os.path.splitext(os.path.basename(path))[0],
        source='deroussi_norre',
        job_length=p['job_length'], op_pt=p['op_pt'],
        time_lag=np.zeros(N), station_cell=station_cell, tau_cells=tau,
        n_vehicles=p['n_vehicles'], veh_start_cell=0, cell_xy=cell_xy,
        synthetic_tau=synthetic,
        notes=dict(n_workstations=n_ws, ws_of_machine=ws_of_machine.tolist()))


def _layout_path_for_machines(nm):
    path = os.path.join(BERT_LAYOUT_DIR, f'layout{nm}.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'no Berterottiere travel layout for {nm} machines ({path}); '
            f'available: {sorted(os.listdir(BERT_LAYOUT_DIR))}')
    return path


def load_berterottiere(instance_path, n_vehicles, layout_path=None,
                       depot_index=0):
    """Load a Berterottiere/Dauzere instance + its travel layout.

    instance_path : Dauzere_Data/Text/{01..18}a.txt (standard FJS format).
    n_vehicles    : fleet size (2/4/6 -- the dpp{k}veh variant).
    layout_path   : (M+1)x(M+1) travel matrix; auto-selected by machine count.
    depot_index   : which location in the matrix is the L/U depot (caveat C3).
    """
    p = parse_fjs_standard(instance_path)
    nm = p['n_machines']
    if layout_path is None:
        layout_path = _layout_path_for_machines(nm)
    tau_full = parse_travel_matrix(layout_path)
    C = tau_full.shape[0]
    if C != nm + 1:
        raise ValueError(f'{layout_path}: {C}x{C} matrix but instance has {nm} '
                         f'machines (expected {nm + 1} locations)')
    if not (0 <= depot_index < C):
        raise ValueError(f'depot_index {depot_index} out of range 0..{C - 1}')
    # Machines occupy every location except the depot, in order. Identity map:
    # machine m (0-based) -> the m-th non-depot cell.
    machine_cells = [c for c in range(C) if c != depot_index]
    station_cell = np.array(machine_cells, dtype=int)
    cell_xy = np.array([[i, 0] for i in range(C)], dtype=float)

    N = int(np.sum(p['job_length']))
    return ExternalInstance(
        name=os.path.splitext(os.path.basename(instance_path))[0],
        source='berterottiere',
        job_length=p['job_length'], op_pt=p['op_pt'],
        time_lag=np.zeros(N), station_cell=station_cell, tau_cells=tau_full,
        n_vehicles=int(n_vehicles), veh_start_cell=depot_index, cell_xy=cell_xy,
        synthetic_tau=False,
        notes=dict(layout=os.path.basename(layout_path), depot_index=depot_index))


# ---------------------------------------------------------------------------
# Convenience: resolve canonical benchmark paths
# ---------------------------------------------------------------------------

def deroussi_norre_path(stem):
    """'fjsp1' | 'fjsp1.txt' -> absolute path under DeroussiNorre/."""
    if not stem.endswith('.txt'):
        stem = stem + '.txt'
    return os.path.join(DN_DIR, stem)


def berterottiere_paths(stem, n_vehicles):
    """'01a' -> (instance_path, layout_path, result_path) for k vehicles.

    stem is the Dauzere id ('01a'..'18a'); the result file is named
    'dpp{stem}_{k}veh.txt' under Berterottiere/dpp{k}veh/."""
    stem = stem.replace('.txt', '')
    instance_path = os.path.join(DAUZERE_DIR, f'{stem}.txt')
    p = parse_fjs_standard(instance_path)
    layout_path = _layout_path_for_machines(p['n_machines'])
    result_path = os.path.join(BERT_RESULT_DIR, f'dpp{n_vehicles}veh',
                               f'dpp{stem}_{n_vehicles}veh.txt')
    return instance_path, layout_path, result_path


def load_berterottiere_by_stem(stem, n_vehicles, depot_index=0):
    """Load '01a' at k vehicles; also returns the published best-known Cmax if
    the result file exists (else None). -> (ExternalInstance, best_known)."""
    instance_path, layout_path, result_path = berterottiere_paths(stem, n_vehicles)
    inst = load_berterottiere(instance_path, n_vehicles, layout_path=layout_path,
                              depot_index=depot_index)
    best_known = parse_berterottiere_cmax(result_path) \
        if os.path.exists(result_path) else None
    return inst, best_known
