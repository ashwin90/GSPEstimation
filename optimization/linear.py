import sys

import cplex

from utils import LINEAR_SOLVER_TIME_LIMIT
from utils import mprint

cpx = cplex.Cplex()
cpx.parameters.workmem.set(8192)  # 8 GB

class LinearProblem(object):
    def amount_of_variables(self):
        raise NotImplementedError('Subclass responsibility')

    def objective_coefficients(self):
        raise NotImplementedError('Subclass responsibility')

    def lower_bounds(self):
        raise NotImplementedError('Subclass responsibility')

    def upper_bounds(self):
        raise NotImplementedError('Subclass responsibility')

    def variable_types(self):
        raise NotImplementedError('Subclass responsibility')

    def variable_names(self):
        raise NotImplementedError('Subclass responsibility')

    def constraints(self):
        raise NotImplementedError('Subclass responsibility')


class Constraints(object):
    def constraints(self):
        raise NotImplementedError('Subclass responsibility')


class LinearSolver(object):
    def solve(self, linear_problem, profiler, prev_soln_provider=None):
        problem = cplex.Cplex()

        problem.parameters.timelimit.set(LINEAR_SOLVER_TIME_LIMIT)
        # problem.parameters.emphasis.mip.set(1)
        # problem.parameters.mip.strategy.probe.set(3)

        problem.set_log_stream(None)
        problem.set_error_stream(sys.stderr)
        problem.set_warning_stream(sys.stdout)
        problem.set_results_stream(None)

        problem.objective.set_sense(problem.objective.sense.maximize)

        problem.variables.add(obj=linear_problem.objective_coefficients(),
                              lb=linear_problem.lower_bounds(),
                              ub=linear_problem.upper_bounds(),
                              types=linear_problem.variable_types(),
                              names=linear_problem.variable_names())

        problem.linear_constraints.add(lin_expr=linear_problem.constraints()['linear_expressions'],
                                       senses=''.join(linear_problem.constraints()['senses']),
                                       rhs=linear_problem.constraints()['independent_terms'],
                                       names=linear_problem.constraints()['names'])

        if prev_soln_provider:
            problem.start.set_start([], [], prev_soln_provider, [], [], [])

        problem.solve()

        mprint('')
        ip_soln_status = problem.solution.get_status()
        # if ip_soln_status == 107 or ip_soln_status == 108:
        mprint('MIP Finished: %s' % problem.solution.get_status_string())
        mprint('')

        return problem.solution.get_objective_value(), lambda l: problem.solution.get_values(l)
