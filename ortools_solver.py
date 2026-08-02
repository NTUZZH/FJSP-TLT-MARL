from ortools.sat.python import cp_model
import numpy as np
import time
import os
from tqdm import tqdm
import sys
from params import configs
from data_utils import pack_data_from_config
import collections

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


def solve_instances(config):
    """
        Solve 'test_data' from 'data_source' using OR-Tools
        with time limits 'max_solve_time' for each instance,
        and save the result to './or_solution/{data_source}'
    :param config: a package of parameters
    :return:
    """
    # p = psutil.Process()
    # p.cpu_affinity(range(config.low, config.high))

    if not os.path.exists(f'./or_solution/{config.data_source}'):
        os.makedirs(f'./or_solution/{config.data_source}')

    data_list = pack_data_from_config(config.data_source, config.test_data)

    save_direc = f'./or_solution/{config.data_source}'
    if not os.path.exists(save_direc):
        os.makedirs(save_direc)

    for data in data_list:
        dataset = data[0]
        data_name = data[1]
        save_path = save_direc + f'/solution_{data_name}.npy'
        save_subpath = save_direc + f'/{data_name}'

        if not os.path.exists(save_subpath):
            os.makedirs(save_subpath)

        if (not os.path.exists(save_path)) or config.cover_flag:
            print("-" * 25 + "Solve Setting" + "-" * 25)
            print(f"solve data name : {data_name}")
            print(f"path : ./data/{config.data_source}/{data_name}")

            # search for the start index
            for root, dirs, files in os.walk(save_subpath):
                index = len([int(f.split("_")[-1][:-4]) for f in files])

            print(f"left instances: dataset[{index}, {len(dataset[0])})")
            for k in tqdm(range(index, len(dataset[0])), file=sys.stdout, desc="progress", colour='blue'):
                jobs, num_machines = matrix_to_the_format_for_solving(dataset[0][k], dataset[1][k])
                solution, solveTime = fjsp_solver(jobs=jobs,
                                                  num_machines=num_machines,
                                                  time_limits=config.max_solve_time)
                tqdm.write(
                    f"Instance {k + 1}, solution:{solution}, solveTime:{solveTime}, systemtime:{time.strftime('%m-%d %H:%M:%S')}")
                np.save(save_subpath + f'/solution_{data_name}_{str.zfill(str(k + 1), 3)}.npy',
                        np.array([solution, solveTime]))

            print("load results...")
            results = []
            for i in range(len(dataset[0])):
                solve_msg = np.load(save_subpath + f'/solution_{data_name}_{str.zfill(str(i + 1), 3)}.npy')
                results.append(solve_msg)

            np.save(save_path, np.array(results))
            print("successfully save results...")


def matrix_to_the_format_for_solving(job_length, op_pt):
    """
        Convert matrix form of the data into the format needed by OR-Tools
    :param job_length: the number of operations in each job (shape [J])
    :param op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    :return:
    """
    num_ops, num_machines = op_pt.shape
    num_jobs = job_length.shape[0]
    jobs = []
    op_idx = 0
    for j in range(num_jobs):
        job_msg = []
        for k in range(job_length[j]):
            able_mchs = np.where(op_pt[op_idx] != 0)[0]
            op_msg = [(op_pt[op_idx, k], k) for k in able_mchs]
            job_msg.append(op_msg)
            op_idx += 1
        jobs.append(job_msg)
    return jobs, num_machines


def fjsp_solver(jobs, num_machines, time_limits, time_lag=None, return_schedule=False,
               warmstart=None):
    """
        solve a fjsp instance by OR-Tools
        (imported from https://github.com/google/or-tools/blob/master/examples/python/flexible_job_shop_sat.py)
    :param jobs: a list of processing information
        eg. jobs = [  # task = (processing_time, machine_id)
                        [  # Job 0
                            [(3, 0), (1, 1), (5, 2)],  # task 0 with 3 alternatives
                            [(2, 0), (4, 1), (6, 2)],  # task 1 with 3 alternatives
                            [(2, 0), (3, 1), (1, 2)],  # task 2 with 3 alternatives
                        ],
                        [  # Job 1
                            [(2, 0), (3, 1), (4, 2)],
                            [(1, 0), (5, 1), (4, 2)],
                            [(2, 0), (1, 1), (4, 2)],
                        ],
                        [  # Job 2
                            [(2, 0), (1, 1), (4, 2)],
                            [(2, 0), (3, 1), (4, 2)],
                            [(3, 0), (1, 1), (5, 2)],
                        ],
                    ]
    :param num_machines: the number of machines
    :param time_limits: the time limits for solving the instance
    :param time_lag: optional post-operation time-lag array (shape [N], flat, ops
                in job-by-job precedence order as in 'jobs'). lag[i] is a mandatory
                wait AFTER op i completes before its job successor may start; the
                machine is free during the lag. None or all-zeros => identical to
                the original lag-free behavior.
    :param return_schedule: if True, also return the per-op (assigned_mch, op_ct)
                schedule (flat arrays [N] in the same op order as time_lag), so the
                solution can be reconstructed / validated. Off by default to keep
                existing callers untouched.
    :return: (objective_value, solve_time) when return_schedule is False (default);
             (objective_value, solve_time, status_name, assigned_mch, op_ct)
             when return_schedule is True.
    """

    num_jobs = len(jobs)
    all_jobs = range(num_jobs)

    all_machines = range(num_machines)

    # Flatten the lags into a global op counter aligned with the job-by-job order.
    num_ops = sum(len(job) for job in jobs)
    if time_lag is None:
        lag = [0] * num_ops
    else:
        lag = [int(x) for x in time_lag]
        assert len(lag) == num_ops, \
            f"time_lag length {len(lag)} != number of ops {num_ops}"

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()

    horizon = 0
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration
    # Lags consume wall-clock between ops, so widen the horizon by the total lag
    # (sum of all lags is a safe upper bound). No-op when there are no lags.
    horizon += sum(lag)

    # print('Horizon = %i' % horizon)

    # Global storage of variables.
    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    # Scan the jobs and create the relevant variables and intervals.
    op_idx = 0  # global op index (job-by-job precedence order), aligns with 'lag'
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        previous_lag = 0
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            # Add precedence with previous task in the same job.
            # With a post-op lag the successor is blocked until the predecessor's
            # end plus its lag (machine free during the lag); previous_lag==0
            # reduces this to the original start >= previous_end.
            if previous_end is not None:
                model.Add(start >= previous_end + previous_lag)
            previous_end = end
            previous_lag = lag[op_idx]
            op_idx += 1

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, 'start' + alt_suffix)
                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(0, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)

                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Optional warm-start: seed the search with a feasible heuristic schedule
    # (flat per-op assigned machine + completion time, job-by-job order). Hints
    # only guide the solver's search; they do NOT change the model or objective,
    # so this is the SAME optimization with a better first incumbent. No-op when
    # warmstart is None (existing callers are byte-identical).
    if warmstart is not None:
        ws_mch, ws_ct = warmstart
        op_idx = 0
        for job_id in all_jobs:
            for task_id in range(len(jobs[job_id])):
                alts = jobs[job_id][task_id]
                mch = int(ws_mch[op_idx])
                dur, chosen_alt = 0, None
                for alt_id, (d, m) in enumerate(alts):
                    if m == mch:
                        dur, chosen_alt = d, alt_id
                st = max(0, int(ws_ct[op_idx]) - dur)
                model.AddHint(starts[(job_id, task_id)], st)
                if len(alts) > 1 and chosen_alt is not None:
                    for alt_id in range(len(alts)):
                        model.AddHint(presences[(job_id, task_id, alt_id)],
                                      1 if alt_id == chosen_alt else 0)
                op_idx += 1

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limits
    solution_printer = SolutionPrinter()

    total1 = time.time()
    status = solver.Solve(model, solution_printer)
    total2 = time.time()

    if not return_schedule:
        return solver.ObjectiveValue(), total2 - total1

    # Reconstruct the schedule: for each op recover the selected machine and its
    # completion time (start + chosen processing time). Flat arrays in the same
    # job-by-job op order as 'lag' / the generator's op_pt rows.
    assigned_mch = [-1] * num_ops
    op_ct = [0] * num_ops
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        op_idx = 0
        for job_id in all_jobs:
            for task_id in range(len(jobs[job_id])):
                start_value = solver.Value(starts[(job_id, task_id)])
                machine = -1
                duration = 0
                for alt_id in range(len(jobs[job_id][task_id])):
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        duration = jobs[job_id][task_id][alt_id][0]
                        machine = jobs[job_id][task_id][alt_id][1]
                assigned_mch[op_idx] = machine
                op_ct[op_idx] = start_value + duration
                op_idx += 1

    return (solver.ObjectiveValue(), total2 - total1, solver.StatusName(status),
            np.array(assigned_mch, dtype=int), np.array(op_ct, dtype=int))

    # Print final solution.
    # for job_id in all_jobs:
    #     print('Job %i:' % job_id)
    #     for task_id in range(len(jobs[job_id])):
    #         start_value = solver.Value(starts[(job_id, task_id)])
    #         machine = -1
    #         duration = -1
    #         selected = -1
    #         for alt_id in range(len(jobs[job_id][task_id])):
    #             if solver.Value(presences[(job_id, task_id, alt_id)]):
    #                 duration = jobs[job_id][task_id][alt_id][0]
    #                 machine = jobs[job_id][task_id][alt_id][1]
    #                 selected = alt_id
    #         print(
    #             '  task_%i_%i starts at %i (alt %i, machine %i, duration %i)' %
    #             (job_id, task_id, start_value, selected, machine, duration))
    #
    # print('Solve status: %s' % solver.StatusName(status))
    # print('Optimal objective value: %i' % solver.ObjectiveValue())
    # print('Statistics')
    # print('  - conflicts : %i' % solver.NumConflicts())
    # print('  - branches  : %i' % solver.NumBranches())
    # print('  - wall time : %f s' % solver.WallTime())


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """
        Print intermediate solutions.
    """

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """
            Called at each new solution.
        """
        # print('Solution %i, time = %f s, objective = %i' %
        #       (self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


if __name__ == '__main__':
    solve_instances(config=configs)
