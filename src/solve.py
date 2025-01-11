import numpy as np
from opt_problem import optProblem

class solveProblem():
    def __init__(self):

        self.opt = optProblem()
        self.max_iter = 15
        self.converged_iter = self.max_iter
        self.sigma_list = np.empty((self.max_iter, 1))
        self.trajectory_list = np.zeros((self.max_iter, 14, self.opt.nk))

        self.x  = np.zeros((14, self.opt.nk))
        self.u = np.zeros((3, self.opt.nk))
        self.sigma = 0
        self.cost = 0
        self.nu = np.zeros((14, self.opt.nk-1))

    def solve(self):
        for i in range(0, self.max_iter):
            self.opt.discretize()
            self.x, self.u, self.sigma, self.cost, self.nu = self.opt.solve_cvx_problem()

            self.sigma_list[i] = self.sigma
            self.trajectory_list[i, :, :] = self.x

            if np.abs(self.sigma - self.cost) <= 0.0005:
                self.converged_iter = i
                break
        return 